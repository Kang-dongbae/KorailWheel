import os, glob, csv, random, re
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import cv2
from pathlib import Path
import config as cfg

# =========================
# 1) Helpers
# =========================
def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: str, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def resize_hw_fn(rgb: np.ndarray, hw: Tuple[int,int]) -> np.ndarray:
    H, W = hw
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def gaussian_1d_smooth(x: np.ndarray, k: int = 11) -> np.ndarray:
    k = int(k)
    if k < 3: return x
    if k % 2 == 0: k += 1
    xx = x.astype(np.float32)[None, :]
    yy = cv2.GaussianBlur(xx, (k, 1), 0)
    return yy[0]

def robust_polyfit(y: np.ndarray, x: np.ndarray, deg: int = 1, iters: int = 3, resid_thresh: float = 4.0) -> np.ndarray:
    mask = np.isfinite(x)
    yv, xv = y[mask], x[mask]
    if len(xv) < deg + 2:
        return np.array([0.0, float(np.nanmedian(xv))]) if len(xv) else np.array([0.0, 0.0])

    coef = np.polyfit(yv, xv, deg)
    for _ in range(iters):
        pred = np.polyval(coef, yv)
        resid = np.abs(pred - xv)
        inl = resid < resid_thresh
        if inl.sum() < max(deg + 2, int(0.5 * len(xv))): break
        coef = np.polyfit(yv[inl], xv[inl], deg)
        yv, xv = yv[inl], xv[inl]
    return coef

# =========================
# 2) Edge detection + Logic
# =========================
@dataclass
class EdgeDebug:
    left0: int; right0: int
    valid_rate: float; mean_strength: float
    width_mean: float; width_std: float
    slope_left: float; slope_right: float
    edge_ok: bool; reason: str

def detect_edges_per_row(rgb_rs: np.ndarray, win: int = 20, min_width: int = 30, grad_thr: float = 8.0):
    gray = to_gray(rgb_rs)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    sobx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    g = np.abs(sobx)
    H, W = g.shape
    col_prof = gaussian_1d_smooth(g.mean(axis=0), k=21)
    mid = W // 2
    left0 = int(np.argmax(col_prof[:mid]))
    right0 = int(np.argmax(col_prof[mid:]) + mid)

    xL = np.full((H,), np.nan, dtype=np.float32)
    xR = np.full((H,), np.nan, dtype=np.float32)
    sL = np.zeros((H,), dtype=np.float32)
    sR = np.zeros((H,), dtype=np.float32)

    for y in range(H):
        row = g[y]
        l1, l2 = max(0, left0 - win), min(W, left0 + win + 1)
        r1, r2 = max(0, right0 - win), min(W, right0 + win + 1)
        li = int(np.argmax(row[l1:l2]) + l1)
        ri = int(np.argmax(row[r1:r2]) + r1)
        sl, sr = float(row[li]), float(row[ri])
        if sl >= grad_thr and sr >= grad_thr and (ri - li) >= min_width:
            xL[y], xR[y] = li, ri
            sL[y], sR[y] = sl, sr
    return xL, xR, sL, sR, left0, right0

def warp_band_from_lines(rgb_rs: np.ndarray, coefL: np.ndarray, coefR: np.ndarray, band_w: int = 72, inner_margin: float = 2.0):
    H, W = rgb_rs.shape[:2]
    ys = np.arange(H, dtype=np.float32)
    xL = np.clip(np.polyval(coefL, ys).astype(np.float32) + inner_margin, 0, W-2)
    xR = np.clip(np.polyval(coefR, ys).astype(np.float32) - inner_margin, 1, W-1)
    
    bad = (xR - xL) < 10.0
    if np.any(bad): xR[bad] = np.clip(xL[bad] + 10.0, 1, W-1)

    xs = np.linspace(0, 1, band_w, dtype=np.float32)[None, :]
    xmap = xL[:, None] + xs * (xR[:, None] - xL[:, None])
    ymap = np.repeat(ys[:, None], band_w, axis=1)
    
    band = cv2.remap(rgb_rs, xmap, ymap, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return band, xL, xR

def compute_qc(xL, xR, sL, sR, left0, right0, coefL, coefR):
    valid = np.isfinite(xL) & np.isfinite(xR)
    vr = float(valid.mean()) if len(valid) else 0.0
    if valid.sum() > 0:
        widths = (xR[valid] - xL[valid]).astype(np.float32)
        w_mean, w_std = float(widths.mean()), float(widths.std())
        w_cv = float(w_std / (w_mean + 1e-6))
        ms = float(((sL[valid] + sR[valid]) * 0.5).mean())
    else:
        w_mean, w_std, w_cv, ms = 0.0, 0.0, 1e9, 0.0

    slopeL = float(coefL[-2]) if len(coefL) >= 2 else 0.0
    slopeR = float(coefR[-2]) if len(coefR) >= 2 else 0.0

    edge_ok = True
    reason = "OK"
    if vr < 0.85: edge_ok, reason = False, f"valid_rate<{0.85}"
    elif ms < 10.0: edge_ok, reason = False, f"mean_strength<{10.0}"
    elif w_cv > 0.15: edge_ok, reason = False, f"width_cv>{0.15}" # notebook3 값 반영
    elif (abs(slopeL) > 0.25) or (abs(slopeR) > 0.25): edge_ok, reason = False, "abs(slope)>0.25"

    return EdgeDebug(left0, right0, vr, ms, w_mean, w_std, slopeL, slopeR, edge_ok, reason)

def extract_band_v2(rgb: np.ndarray, resize_hw=(512,256), band_w=72):
    rgb_rs = resize_hw_fn(rgb, resize_hw)
    xL, xR, sL, sR, left0, right0 = detect_edges_per_row(rgb_rs, win=22, min_width=28, grad_thr=8.0)
    H = rgb_rs.shape[0]
    ys = np.arange(H, dtype=np.float32)
    coefL = robust_polyfit(ys, xL, deg=1, resid_thresh=5.0)
    coefR = robust_polyfit(ys, xR, deg=1, resid_thresh=5.0)
    
    qc = compute_qc(xL, xR, sL, sR, left0, right0, coefL, coefR)
    band, xL_fit, xR_fit = warp_band_from_lines(rgb_rs, coefL, coefR, band_w=band_w)
    
    ov = rgb_rs.copy()
    # Debug overlay (simplified)
    for y in range(0, H, 2):
        cv2.circle(ov, (int(xL_fit[y]), y), 1, (255,0,0), -1)
        cv2.circle(ov, (int(xR_fit[y]), y), 1, (0,255,0), -1)
        
    return band, qc, ov

# =========================
# 3) Process Routines
# =========================
def process_split(split: str):
    in_files = sorted(glob.glob(os.path.join(cfg.RAW_DATA_ROOT, split, "images", "*.jpg")))
    print(f"[{split}] Input: {len(in_files)}")
    
    pass_dir = os.path.join(cfg.OUT_BAND_ROOT, split, "images")
    rej_dir = os.path.join(cfg.OUT_BAND_ROOT, split, "reject", "images")
    dbg_dir = os.path.join(cfg.OUT_BAND_ROOT, split, "debug")
    cfg.ensure_dir(pass_dir); cfg.ensure_dir(rej_dir); cfg.ensure_dir(dbg_dir)

    qc_rows = []
    saved_dbg = 0
    for fp in in_files:
        name = Path(fp).name
        rgb = imread_rgb(fp)
        band, qc, ov = extract_band_v2(rgb, resize_hw=cfg.RESIZE_HW, band_w=cfg.BAND_W)
        
        out_fp = os.path.join(pass_dir if qc.edge_ok else rej_dir, name)
        imwrite_rgb(out_fp, band)

        if not qc.edge_ok and saved_dbg < 200:
            rgb_rs = resize_hw_fn(rgb, cfg.RESIZE_HW)
            H, W = rgb_rs.shape[:2]
            canvas = np.zeros((H, W + W + cfg.BAND_W, 3), dtype=np.uint8)
            canvas[:, :W] = rgb_rs; canvas[:, W:W+W] = ov; canvas[:, W+W:] = band
            imwrite_rgb(os.path.join(dbg_dir, f"{Path(fp).stem}__{qc.reason}.png"), canvas)
            saved_dbg += 1
        
        qc_rows.append({"file": name, "edge_ok": int(qc.edge_ok), "reason": qc.reason})

    # Save CSV
    csv_path = os.path.join(cfg.OUT_BAND_ROOT, split, "qc_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qc_rows[0].keys())
        w.writeheader()
        w.writerows(qc_rows)
    print(f"[{split}] QC Pass: {sum(r['edge_ok'] for r in qc_rows)}/{len(qc_rows)}")

def process_tiling_split(split: str):
    in_dir = os.path.join(cfg.OUT_BAND_ROOT, split, "images")
    files = sorted(glob.glob(os.path.join(in_dir, "*.jpg")))
    out_dir = os.path.join(cfg.OUT_TILE_ROOT, split, "images")
    cfg.ensure_dir(out_dir)
    print(f"[Tile {split}] Targets: {len(files)}")

    for fp in files:
        band = imread_rgb(fp)
        band = resize_hw_fn(band, (cfg.RESIZE_HW[0], cfg.BAND_W))
        H, W = band.shape[:2]
        k = 0
        stem = Path(fp).stem
        for y0 in range(0, H - cfg.TILE_H + 1, cfg.STRIDE):
            y1 = y0 + cfg.TILE_H
            tile = band[y0:y1, :, :]
            imwrite_rgb(os.path.join(out_dir, f"{stem}__y{y0:04d}-{y1:04d}__k{k:02d}.jpg"), tile)
            k += 1

if __name__ == "__main__":
    for sp in ["train", "valid", "test"]:
        process_split(sp)
    for sp in ["train", "valid", "test"]:
        process_tiling_split(sp)
    print("Preprocessing Done.")