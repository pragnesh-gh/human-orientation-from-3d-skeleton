#!/usr/bin/env python3
"""
Depth anchoring visualization (paper-friendly figure)

Given:
- RGB image (color_XXXXX.png)
- Aligned depth image (depth_XXXXX.png, 16-bit)
- keypoints.npz from PoseFormerV2 demo output (contains reconstruction)
This script:
- Reads pelvis pixel (u,v) from keypoints.npz (joint index 0)
- Draws pelvis marker on RGB
- Draws a fixed-size window (e.g., 7x7) on depth centered at (u,v)
- Computes median depth (ignoring zeros), converts to meters
- Backprojects (u,v,Z) into camera frame using intrinsics (fx,fy,cx,cy)
- Saves a single IEEE-suitable figure (RGB + Depth + numeric values)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Defaults (override via CLI)
# -----------------------------
# If these are not correct for your camera, override using --fx --fy --cx --cy
DEFAULT_FX = 911.47
DEFAULT_FY = 911.56
DEFAULT_CX = 654.27
DEFAULT_CY = 366.90

# Typical aligned depth PNG is in millimeters (RealSense-style): meters = raw * 0.001
DEFAULT_DEPTH_SCALE = 0.001

DEFAULT_WINDOW_SIZE = 7  # must be odd for symmetric window


@dataclass
class BackprojResult:
    u: int
    v: int
    window_size: int
    valid_count: int
    z_raw_median: Optional[int]
    z_m: Optional[float]
    x_m: Optional[float]
    y_m: Optional[float]


def parse_index_from_name(p: str) -> Optional[int]:
    """
    Extract trailing integer from e.g. color_000563.png or depth_000563.png.
    Returns int(563) or None if no match.
    """
    m = re.search(r"(\d+)(?=\D*$)", Path(p).stem)  # last number in stem
    return int(m.group(1)) if m else None


def load_rgb(rgb_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_depth(depth_path: Path) -> np.ndarray:
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    if d.ndim != 2:
        raise ValueError(f"Depth must be single-channel (H,W). Got shape={d.shape} dtype={d.dtype}")
    # Keep as integer; common: uint16
    return d


def load_pelvis_uv_from_npz(npz_path: Path, frame_idx: int) -> Tuple[float, float]:
    """
    Reads pelvis (joint 0) pixel (u,v) from keypoints.npz.

    Expected:
      reconstruction: (1, T, 17, 2) or (T, 17, 2)
    """
    data = np.load(str(npz_path), allow_pickle=True)
    if "reconstruction" not in data:
        raise KeyError(f"NPZ missing 'reconstruction': {npz_path}")

    rec = data["reconstruction"]
    if rec.ndim == 4:
        # (1, T, 17, 2)
        rec0 = rec[0]
    elif rec.ndim == 3:
        # (T, 17, 2)
        rec0 = rec
    else:
        raise ValueError(f"Unexpected reconstruction shape: {rec.shape}")

    T = rec0.shape[0]
    if T == 1:
        fi = 0
    else:
        fi = int(np.clip(frame_idx, 0, T - 1))

    pelvis = rec0[fi, 0]  # joint index 0 (pelvis/root) in your H36M-17 pipeline
    u, v = float(pelvis[0]), float(pelvis[1])
    return u, v


def median_depth_in_window(depth: np.ndarray, u: int, v: int, window_size: int) -> Tuple[Optional[int], int, Tuple[int, int, int, int]]:
    """
    Returns:
      z_raw_median (int) or None if no valid depth
      valid_count (# of nonzero pixels in window)
      window bbox (x0,y0,x1,y1) inclusive-exclusive, clipped to image bounds
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd (e.g., 7)")

    H, W = depth.shape[:2]
    r = window_size // 2
    x0 = max(0, u - r)
    y0 = max(0, v - r)
    x1 = min(W, u + r + 1)
    y1 = min(H, v + r + 1)

    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0]
    valid_count = int(valid.size)

    if valid_count == 0:
        return None, 0, (x0, y0, x1, y1)

    z_raw = int(np.median(valid).item())
    return z_raw, valid_count, (x0, y0, x1, y1)


def backproject(u: int, v: int, z_raw_median: Optional[int], valid_count: int,
                fx: float, fy: float, cx: float, cy: float, depth_scale: float,
                window_size: int) -> BackprojResult:
    if z_raw_median is None or valid_count == 0:
        return BackprojResult(u=u, v=v, window_size=window_size, valid_count=valid_count,
                              z_raw_median=None, z_m=None, x_m=None, y_m=None)

    z_m = float(z_raw_median) * float(depth_scale)
    x_m = (float(u) - float(cx)) / float(fx) * z_m
    y_m = (float(v) - float(cy)) / float(fy) * z_m
    return BackprojResult(u=u, v=v, window_size=window_size, valid_count=valid_count,
                          z_raw_median=z_raw_median, z_m=z_m, x_m=x_m, y_m=y_m)


def normalize_depth_for_display(depth: np.ndarray) -> np.ndarray:
    """
    Paper-friendly depth visualization:
    - ignore zeros
    - scale using robust percentiles (1..99)
    Returns float image in [0,1].
    """
    d = depth.astype(np.float32)
    valid = d[d > 0]
    if valid.size == 0:
        return np.zeros_like(d, dtype=np.float32)

    lo = np.percentile(valid, 1)
    hi = np.percentile(valid, 99)
    if hi <= lo:
        hi = lo + 1.0

    dn = (d - lo) / (hi - lo)
    dn = np.clip(dn, 0.0, 1.0)
    dn[d <= 0] = 0.0
    return dn


# def make_figure(rgb: np.ndarray, depth: np.ndarray,
#                 rgb_path: Path, depth_path: Path,
#                 res: BackprojResult, win_bbox: Tuple[int, int, int, int],
#                 fx: float, fy: float, cx: float, cy: float, depth_scale: float,
#                 out_path: Path) -> None:
#     # --- Style tweaks for paper look ---
#     plt.rcParams.update({
#         "font.family": "serif",
#         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
#         "font.size": 9,
#         "axes.titlesize": 10,
#     })
#
#     depth_vis = normalize_depth_for_display(depth)
#
#     fig = plt.figure(figsize=(7.0, 3.4), dpi=450)
#     gs = fig.add_gridspec(1, 2, wspace=0.03)
#
#     ax_rgb = fig.add_subplot(gs[0, 0])
#     ax_dep = fig.add_subplot(gs[0, 1])
#
#     # ---------------- RGB panel ----------------
#     ax_rgb.imshow(rgb)
#     ax_rgb.set_title("RGB (pelvis anchor)")
#     ax_rgb.axis("off")
#
#     # clean pelvis marker
#     ax_rgb.scatter([res.u], [res.v], s=40, c="red", edgecolors="white", linewidths=1.0, zorder=5)
#     ax_rgb.text(res.u + 8, res.v - 10, f"({res.u}, {res.v})",
#                 color="white", fontsize=9,
#                 bbox=dict(boxstyle="round,pad=0.20", fc=(0, 0, 0, 0.55), ec="none"),
#                 zorder=6)
#
#     # ---------------- Depth panel ----------------
#     im = ax_dep.imshow(depth_vis, cmap="magma", vmin=0.0, vmax=1.0)
#     ax_dep.set_title(f"Aligned depth (window {res.window_size}×{res.window_size})")
#     ax_dep.axis("off")
#
#     x0, y0, x1, y1 = win_bbox
#     # window rectangle
#     ax_dep.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=1.6, color="cyan")
#     # center marker
#     ax_dep.scatter([res.u], [res.v], s=45, c="cyan", edgecolors="black", linewidths=0.6, zorder=5)
#
#     # ---- Zoom inset on depth window (makes it pop) ----
#     # compute inset crop around the window
#     pad = 25
#     H, W = depth.shape[:2]
#     ix0 = max(0, x0 - pad)
#     iy0 = max(0, y0 - pad)
#     ix1 = min(W, x1 + pad)
#     iy1 = min(H, y1 + pad)
#
#     inset = ax_dep.inset_axes([0.62, 0.62, 0.36, 0.36])  # [x,y,w,h] in axes fraction
#     inset.imshow(depth_vis[iy0:iy1, ix0:ix1], cmap="magma", vmin=0.0, vmax=1.0)
#     inset.axis("off")
#     # draw window rectangle inside inset (offset coords)
#     inset.plot([x0 - ix0, x1 - ix0, x1 - ix0, x0 - ix0, x0 - ix0],
#                [y0 - iy0, y0 - iy0, y1 - iy0, y1 - iy0, y0 - iy0],
#                linewidth=1.4, color="cyan")
#     inset.scatter([res.u - ix0], [res.v - iy0], s=35, c="cyan", edgecolors="black", linewidths=0.5)
#
#     # ---------------- Compact stats (top-left overlay) ----------------
#     z_raw_str = "NA" if res.z_raw_median is None else str(res.z_raw_median)
#     z_m_str = "NA" if res.z_m is None else f"{res.z_m:.3f}"
#     x_str = "NA" if res.x_m is None else f"{res.x_m:.3f}"
#     y_str = "NA" if res.y_m is None else f"{res.y_m:.3f}"
#     z_str = "NA" if res.z_m is None else f"{res.z_m:.3f}"
#
#     stats = (
#         "Metric translation from depth\n"
#         f"Z_raw: {z_raw_str}   Z: {z_m_str} m   valid: {res.valid_count}\n"
#         f"(X,Y,Z): ({x_str}, {y_str}, {z_str}) m\n"
#         f"fx={fx:g}, fy={fy:g}, cx={cx:g}, cy={cy:g}   scale={depth_scale:g}"
#     )
#
#     # place in RGB panel for compactness
#     ax_rgb.text(0.01, 0.01, stats, transform=ax_rgb.transAxes,
#                 va="bottom", ha="left", fontsize=8,
#                 bbox=dict(boxstyle="round,pad=0.25", fc=(1, 1, 1, 0.90), ec="none"))
#
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.02)
#     plt.close(fig)

def make_figure(rgb: np.ndarray, depth: np.ndarray,
                rgb_path: Path, depth_path: Path,
                res: BackprojResult, win_bbox: Tuple[int, int, int, int],
                fx: float, fy: float, cx: float, cy: float, depth_scale: float,
                out_path: Path) -> None:
    # ---- Paper-style fonts ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
    })

    depth_vis = normalize_depth_for_display(depth)

    # Vertical layout: RGB on top, Depth below
    fig = plt.figure(figsize=(4.8, 6.4), dpi=450)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.06)

    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_dep = fig.add_subplot(gs[1, 0])

    # ================= RGB panel =================
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB (pelvis anchor)")
    ax_rgb.axis("off")

    # Pelvis marker
    ax_rgb.scatter(
        [res.u], [res.v],
        s=45, c="red",
        edgecolors="white", linewidths=1.0, zorder=5
    )
    ax_rgb.text(
        res.u + 8, res.v - 10, f"({res.u}, {res.v})",
        color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.20",
                  fc=(0, 0, 0, 0.55), ec="none"),
        zorder=6
    )

    # Compact stats box (bottom-left of RGB)
    z_raw_str = "NA" if res.z_raw_median is None else str(res.z_raw_median)
    z_m_str = "NA" if res.z_m is None else f"{res.z_m:.3f}"
    x_str = "NA" if res.x_m is None else f"{res.x_m:.3f}"
    y_str = "NA" if res.y_m is None else f"{res.y_m:.3f}"
    z_str = "NA" if res.z_m is None else f"{res.z_m:.3f}"

    stats = (
        "Metric translation from depth\n"
        f"Z_raw: {z_raw_str}   Z: {z_m_str} m   valid: {res.valid_count}\n"
        f"(X,Y,Z): ({x_str}, {y_str}, {z_str}) m\n"
        f"fx={fx:g}, fy={fy:g}, cx={cx:g}, cy={cy:g}"
    )

    ax_rgb.text(
        0.01, 0.01, stats,
        transform=ax_rgb.transAxes,
        va="bottom", ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25",
                  fc=(1, 1, 1, 0.90), ec="none")
    )

    # ================= Depth panel =================
    ax_dep.imshow(depth_vis, cmap="magma", vmin=0.0, vmax=1.0)
    ax_dep.set_title(f"Aligned depth (pseudo-color, window {res.window_size}×{res.window_size})")
    ax_dep.axis("off")

    x0, y0, x1, y1 = win_bbox

    # Sampling window
    ax_dep.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        linewidth=1.6, color="cyan"
    )
    ax_dep.scatter(
        [res.u], [res.v],
        s=45, c="cyan",
        edgecolors="black", linewidths=0.6, zorder=5
    )

    # ---- Zoom inset on depth window ----
    pad = 25
    H, W = depth.shape[:2]
    ix0 = max(0, x0 - pad)
    iy0 = max(0, y0 - pad)
    ix1 = min(W, x1 + pad)
    iy1 = min(H, y1 + pad)

    inset = ax_dep.inset_axes([0.62, 0.58, 0.35, 0.35])
    inset.imshow(
        depth_vis[iy0:iy1, ix0:ix1],
        cmap="magma", vmin=0.0, vmax=1.0
    )
    inset.axis("off")

    inset.plot(
        [x0 - ix0, x1 - ix0, x1 - ix0, x0 - ix0, x0 - ix0],
        [y0 - iy0, y0 - iy0, y1 - iy0, y1 - iy0, y0 - iy0],
        linewidth=1.4, color="cyan"
    )
    inset.scatter(
        [res.u - ix0], [res.v - iy0],
        s=35, c="cyan",
        edgecolors="black", linewidths=0.5
    )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=str, help="Path to input_2D/keypoints.npz")
    ap.add_argument("--rgb", required=True, type=str, help="Path to color_XXXXX.png")
    ap.add_argument("--depth", required=True, type=str, help="Path to depth_XXXXX.png (aligned, 16-bit)")

    ap.add_argument("--frame-idx", default=None, type=int,
                    help="Frame index inside reconstruction. If omitted, tries to parse from filename.")
    ap.add_argument("--joint-idx", default=0, type=int,
                    help="Joint index for anchoring (default 0 = pelvis/root in your pipeline)")

    ap.add_argument("--window", default=DEFAULT_WINDOW_SIZE, type=int, help="Sampling window size (odd), e.g., 7")
    ap.add_argument("--depth-scale", default=DEFAULT_DEPTH_SCALE, type=float, help="Meters per raw depth unit")
    ap.add_argument("--fx", default=DEFAULT_FX, type=float)
    ap.add_argument("--fy", default=DEFAULT_FY, type=float)
    ap.add_argument("--cx", default=DEFAULT_CX, type=float)
    ap.add_argument("--cy", default=DEFAULT_CY, type=float)

    ap.add_argument("--out", default=None, type=str,
                    help="Output PNG path. Default: <npz_dir>/depth_anchor_<index>.png")

    args = ap.parse_args()

    npz_path = Path(args.npz)
    rgb_path = Path(args.rgb)
    depth_path = Path(args.depth)

    # Determine frame index: CLI > filename > 0
    if args.frame_idx is not None:
        frame_idx = int(args.frame_idx)
    else:
        idx = parse_index_from_name(rgb_path.name) or parse_index_from_name(depth_path.name)
        frame_idx = int(idx) if idx is not None else 0

    # Load inputs
    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)

    # Load keypoints and get pelvis uv
    data = np.load(str(npz_path), allow_pickle=True)
    rec = data["reconstruction"]
    if rec.ndim == 4:
        rec0 = rec[0]  # (T,17,2)
    elif rec.ndim == 3:
        rec0 = rec
    else:
        raise ValueError(f"Unexpected reconstruction shape: {rec.shape}")

    T = rec0.shape[0]
    fi = 0 if T == 1 else int(np.clip(frame_idx, 0, T - 1))

    if args.joint_idx < 0 or args.joint_idx >= rec0.shape[1]:
        raise ValueError(f"joint-idx out of range: {args.joint_idx} (expected 0..{rec0.shape[1]-1})")

    u_f, v_f = float(rec0[fi, args.joint_idx, 0]), float(rec0[fi, args.joint_idx, 1])
    u, v = int(round(u_f)), int(round(v_f))

    # Clip u,v into image bounds (avoid crash near edges)
    H, W = depth.shape[:2]
    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    # Depth median in window
    z_raw_median, valid_count, win_bbox = median_depth_in_window(depth, u, v, args.window)

    # Backproject
    res = backproject(u=u, v=v, z_raw_median=z_raw_median, valid_count=valid_count,
                      fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                      depth_scale=args.depth_scale, window_size=args.window)

    # Output path: same folder as keypoints.npz by default
    if args.out is not None:
        out_path = Path(args.out)
    else:
        name_idx = parse_index_from_name(rgb_path.name)
        suffix = f"{name_idx:06d}" if name_idx is not None else f"{fi:06d}"
        out_path = npz_path.parent / f"depth_anchor_{suffix}.png"

    make_figure(rgb=rgb, depth=depth,
                rgb_path=rgb_path, depth_path=depth_path,
                res=res, win_bbox=win_bbox,
                fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy, depth_scale=args.depth_scale,
                out_path=out_path)

    print(f"[OK] Saved figure: {out_path}")


if __name__ == "__main__":
    main()
