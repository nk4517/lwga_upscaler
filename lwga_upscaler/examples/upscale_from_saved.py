"""
Demo: load saved render gradients and perform upscale.

Usage:
    python -m lwga_upscaler.examples.upscale_from_saved <prefix> [--scale N] [--output path]

Expected files:
    <prefix>_rgb_8bit.png (or _rgb.png for 16-bit)
    <prefix>_dx_8bit.png
    <prefix>_dy_8bit.png
    <prefix>_dxy_8bit.png
    <prefix>_meta.json with {"dx_range": [min, max], "dy_range": [...], "dxy_range": [...]}
"""

import argparse
import json
from pathlib import Path
import numpy as np

import torch
import imageio.v3 as iio

from lwga_upscaler.upscale import gradient_aware_upscale


def load_png(path: Path | str, vmin: float = 0.0, vmax: float = 1.0) -> torch.Tensor:
    """
    Load PNG (auto-detect 8/16 bit) and denormalize to [vmin, vmax].
    Returns: torch.Tensor [H, W, C] float32
    """
    arr = iio.imread(path)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr)
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    return t * (vmax - vmin) + vmin


def save_png(
        tensor_hwc: torch.Tensor,
        path: Path | str,
        bits: int = 16,
        vmin: float | None = None,
        vmax: float | None = None,
) -> tuple[float, float]:
    """
    Save HWC tensor as PNG.
    If vmin/vmax are None, auto-compute from tensor (for signed data).
    Returns: (vmin, vmax) used for normalization.
    """
    t = tensor_hwc.detach().cpu().float()
    if vmin is None or vmax is None:
        vmin = t.min().item()
        vmax = t.max().item()
    t = (t - vmin) / (vmax - vmin + 1e-10)
    t = t.clamp(0, 1)

    if bits == 16:
        arr = (t * 65535).numpy().astype(np.uint16)
    else:
        arr = (t * 255).numpy().astype(np.uint8)

    iio.imwrite(path, arr)
    return vmin, vmax


def main():
    parser = argparse.ArgumentParser(description="Upscale from saved gradient images")
    parser.add_argument("prefix", type=Path, help="Prefix path (e.g. render_grads_1500k)")
    parser.add_argument("--scale", type=float, default=8.0, help="Upscale factor")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    args = parser.parse_args()

    prefix = args.prefix
    with open(f"{prefix}_meta.json") as f:
        meta = json.load(f)

    rgb = load_png(f"{prefix}_rgb.png").cuda()
    dx = load_png(f"{prefix}_dx.png", *meta["dx_range"]).cuda()
    dy = load_png(f"{prefix}_dy.png", *meta["dy_range"]).cuda()
    dxy = load_png(f"{prefix}_dxy.png", *meta["dxy_range"]).cuda()

    h, w, c = rgb.shape
    dst_h, dst_w = round(h * args.scale), round(w * args.scale)
    print(f"Upscaling {w}x{h} -> {dst_w}x{dst_h} (x{args.scale})")

    with torch.no_grad():
        upscaled = gradient_aware_upscale(rgb, dx, dy, dxy, dst_h, dst_w)
        upscaled = torch.clamp(upscaled, 0, 1)

    out_path = args.output or Path(f"{prefix}_x{args.scale:.0f}.png")
    save_png(upscaled, out_path, bits=8, vmin=0.0, vmax=1.0)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
