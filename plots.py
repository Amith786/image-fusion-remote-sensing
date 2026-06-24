"""
plots.py — Save PSNR and SSIM comparison charts as PNG files.

Changed from original
---------------------
- Added  matplotlib.use('Agg')  so the function is safe to call from a
  Flask worker (no GUI / display required).
- Removed plt.show() calls (charts are saved to disk; show() would block
  a headless server indefinitely).
- Function now returns a dict  {'psnr': psnr_path, 'ssim': ssim_path}
  so callers can log or serve the paths without re-computing them.
"""

import os
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt


def plot_graphs(psnr_wavelet, psnr_dl, psnr_final,
                ssim_wavelet, ssim_dl, ssim_final,
                psnr_folder, ssim_folder, timestamp):
    """
    Save PSNR and SSIM bar/line charts to disk.

    Parameters
    ----------
    psnr_wavelet, psnr_dl, psnr_final : float
    ssim_wavelet, ssim_dl, ssim_final : float
    psnr_folder : str  — directory where psnr_<timestamp>.png is saved
    ssim_folder : str  — directory where ssim_<timestamp>.png is saved
    timestamp   : str  — used as filename suffix

    Returns
    -------
    dict  {'psnr': str, 'ssim': str}  — absolute paths to the saved PNGs
    """
    methods      = ['Wavelet', 'DL', 'Adaptive']
    psnr_values  = [psnr_wavelet, psnr_dl, psnr_final]
    ssim_values  = [ssim_wavelet, ssim_dl, ssim_final]

    # ── PSNR chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(methods, psnr_values, marker='o', linewidth=2)
    ax.fill_between(methods, psnr_values, alpha=0.2)

    best_i = psnr_values.index(max(psnr_values))
    ax.scatter(methods[best_i], psnr_values[best_i], s=100, zorder=5)
    ax.text(methods[best_i], psnr_values[best_i], "  Best", fontsize=10)

    ax.set_title("PSNR Trend Across Fusion Methods", fontsize=14)
    ax.set_xlabel("Method")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    psnr_path = os.path.join(psnr_folder, f"psnr_{timestamp}.png")
    fig.savefig(psnr_path)
    plt.close(fig)

    # ── SSIM chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(methods, ssim_values, marker='o', linewidth=2)
    ax.fill_between(methods, ssim_values, alpha=0.2)

    best_i = ssim_values.index(max(ssim_values))
    ax.scatter(methods[best_i], ssim_values[best_i], s=100, zorder=5)
    ax.text(methods[best_i], ssim_values[best_i], "  Best", fontsize=10)

    ax.set_title("SSIM Trend Across Fusion Methods", fontsize=14)
    ax.set_xlabel("Method")
    ax.set_ylabel("SSIM")
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    ssim_path = os.path.join(ssim_folder, f"ssim_{timestamp}.png")
    fig.savefig(ssim_path)
    plt.close(fig)

    print(f"Graphs saved → {psnr_path}, {ssim_path}")
    return {'psnr': psnr_path, 'ssim': ssim_path}
