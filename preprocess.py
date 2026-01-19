# preprocess.py
'''
전체 타일 생성(기본):
python preprocess.py

tiles 모드로 전체:
python preprocess.py tiles

특정 split만(기존 호환):
python preprocess.py external_test

external crop 디버그만:
python preprocess.py external_crops

external tiles만:
python preprocess.py external_tiles

calib crops(선택):
python preprocess.py calib_crops
'''
# preprocess.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from config import ROOT

# -------------------------
# Hyperparams (ONLY here)
# -------------------------
SPLITS = ["train", "valid", "test"]          # 기본
ADD_EXTERNAL = True
EXTERNAL_SPLIT = "external_test"            # ROOT/data/external_test/images, labels

DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "data_tiles"
DBG_DIR  = DATA_DIR / "artifacts" / "debug_preprocess"
DBG_DIR.mkdir(parents=True, exist_ok=True)

# ----- MODE -----
# tiles: train/valid/test 타일 생성 (+ external 있으면 external도 타일 분리 저장)
# external_tiles: external_test만 타일 분리 저장
# external_crops: external_test만 결함 중심 crop 생성(진단/디버그)
# calib_crops: valid에서 normal crop 생성(추후 threshold 캘리브레이션용)
MODE_DEFAULT = "tiles"
MODES = {"tiles", "external_tiles", "external_crops", "calib_crops"}

# External crop params (debug)
CROP_SIZE = 512
N_NORMAL_CROPS_PER_WHEEL = 30
MIN_DEFECT_PIXELS_CROP = 150   # crop 내 defect 픽셀 최소치

# Calib crops params (optional)
CALIB_SPLIT = "valid"
CALIB_N_PER_IMAGE = 30

# Unwrap (rubber-sheet) size
UNWRAP_W = 768
SMOOTH_WIN = 31
Y_MARGIN = 5

# Tiling
TILE = 512
STRIDE = 256
MIN_MASK_COVER = 0.90

# 타일 품질
BLACK_THR = 10
MAX_BLACK_RATIO = 0.15
MIN_TEXTURE_STD = 12.0
MIN_VALID_PIXELS = 5000

# External defect tile labeling
DEFECT_RATIO_THR = 0.02
MIN_DEFECT_PIXELS = 300

# 클래스 id (Roboflow class order에 맞춰야 함)
TREAD_CLS  = 0
DEFECT_CLS = 1

# Optional: Stage1 seg model로 tread 마스크 예측해서 쓰고 싶으면 True
USE_PRED = False
SEG_WEIGHTS = ROOT / "runs" / "tread_seg_stage1" / "weights" / "best.pt"
PRED_IMGSZ = 1024
PRED_CONF = 0.25
PRED_MAXDET = 5
PRED_DEVICE = "0"


# -------------------------
# Utils
# -------------------------
def _img_list(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def _read_yolo_instances(label_path: Path):
    """
    YOLO seg line: cls x y w h x1 y1 x2 y2 ...
    YOLO det line: cls x y w h
    return: list[(cls:int, poly_norm: (N,2) float32)]
    """
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    inst = []
    for line in txt.splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))

        # bbox only
        if len(p) == 5:
            xc, yc, bw, bh = map(float, p[1:5])
            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            inst.append((cls, poly))
        else:
            coords = np.array(list(map(float, p[5:])), dtype=np.float32)
            if coords.size >= 6:
                inst.append((cls, coords.reshape(-1, 2)))
    return inst


def _polys_to_mask(polys_norm, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    if not polys_norm:
        return m
    pts_list = []
    for poly in polys_norm:
        pts = poly.copy()
        pts[:, 0] = np.clip(pts[:, 0] * w, 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1] * h, 0, h - 1)
        pts_list.append(pts.astype(np.int32))
    cv2.fillPoly(m, pts_list, 1)
    return m


def _smooth_1d(x: np.ndarray, win: int):
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, k, mode="valid")


def _mask_clean_roi(mask: np.ndarray):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


def _mask_clean_defect(mask: np.ndarray):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _band_bounds(mask: np.ndarray):
    ys = np.where(mask.sum(axis=1) > 0)[0]
    if ys.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())

    xL = []
    xR = []
    for y in range(y0, y1 + 1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            xL.append(-1); xR.append(-1)
        else:
            xL.append(int(xs.min())); xR.append(int(xs.max()))

    xL = np.array(xL, dtype=np.int32)
    xR = np.array(xR, dtype=np.int32)

    valid = (xL >= 0) & (xR >= 0) & (xR > xL)
    if not valid.any():
        return None

    last = None
    for i in range(len(xL)):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last

    last = None
    for i in range(len(xL) - 1, -1, -1):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last

    xL = _smooth_1d(xL.astype(np.float32), SMOOTH_WIN)
    xR = _smooth_1d(xR.astype(np.float32), SMOOTH_WIN)
    return y0, y1, xL, xR


def _bounds_full(mask_slice: np.ndarray):
    H, W = mask_slice.shape
    xL = np.full(H, -1, dtype=np.int32)
    xR = np.full(H, -1, dtype=np.int32)

    for i in range(H):
        xs = np.where(mask_slice[i] > 0)[0]
        if xs.size:
            xL[i] = int(xs.min())
            xR[i] = int(xs.max())

    valid = (xL >= 0) & (xR > xL)
    if not valid.any():
        return None

    last = None
    for i in range(H):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last

    last = None
    for i in range(H - 1, -1, -1):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last

    xL = _smooth_1d(xL.astype(np.float32), SMOOTH_WIN)
    xR = _smooth_1d(xR.astype(np.float32), SMOOTH_WIN)
    return xL, xR


def _unwrap_rubber_sheet(img: np.ndarray, roi_mask: np.ndarray, extra_masks=None):
    if extra_masks is None:
        extra_masks = []

    h, w = img.shape[:2]
    bb = _band_bounds(roi_mask)
    if bb is None:
        return None
    y0_old, y1_old, _, _ = bb

    y0 = max(0, y0_old - Y_MARGIN)
    y1 = min(h - 1, y1_old + Y_MARGIN)

    roi_slice = roi_mask[y0:y1 + 1, :]
    H = roi_slice.shape[0]
    ys = np.arange(y0, y1 + 1, dtype=np.float32)

    bounds = _bounds_full(roi_slice)
    if bounds is None:
        return None
    xL2, xR2 = bounds

    map_x = np.zeros((H, UNWRAP_W), dtype=np.float32)
    map_y = np.zeros((H, UNWRAP_W), dtype=np.float32)

    for i in range(H):
        xl = float(xL2[i]); xr = float(xR2[i])
        if xr <= xl + 1:
            xr = xl + 1.0
        xs = np.linspace(xl, xr, UNWRAP_W, dtype=np.float32)
        map_x[i, :] = np.clip(xs, 0, w - 1)
        map_y[i, :] = ys[i]

    strip = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    def _remap_mask(m01):
        m = (m01.astype(np.float32) * 255.0)
        s = cv2.remap(
            m, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return (s > 0).astype(np.uint8)

    strip_roi = _remap_mask(roi_mask)
    extra_strips = [_remap_mask(m) for m in extra_masks]
    return strip, strip_roi, extra_strips


def _tile_quality_stats(tile_bgr: np.ndarray, tile_mask01: np.ndarray):
    valid = (tile_mask01 > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0, 1.0, 0.0
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gv = g[valid]
    black_ratio = float((gv < BLACK_THR).mean())
    tex_std = float(gv.std())
    return n_valid, black_ratio, tex_std


def _tile_quality_ok(tile_bgr: np.ndarray, tile_mask01: np.ndarray) -> bool:
    valid_pixels, black_ratio, tex_std = _tile_quality_stats(tile_bgr, tile_mask01)
    if valid_pixels < MIN_VALID_PIXELS:
        return False
    if black_ratio > MAX_BLACK_RATIO:
        return False
    if tex_std < MIN_TEXTURE_STD:
        return False
    return True


def _tile_and_save(strip, roi01, out_img_dir, debug_dir, stem, split, defect01=None, out_def_dir=None, out_norm_dir=None):
    H, W = strip.shape[:2]
    rows = []

    cv2.imwrite(str(debug_dir / f"{stem}_base.png"), strip)

    for y0 in range(0, max(1, H - TILE + 1), STRIDE):
        for x0 in range(0, max(1, W - TILE + 1), STRIDE):
            tile = strip[y0:y0 + TILE, x0:x0 + TILE]
            tm   = roi01[y0:y0 + TILE, x0:x0 + TILE]

            if tile.shape[0] != TILE or tile.shape[1] != TILE:
                pad_h = TILE - tile.shape[0]
                pad_w = TILE - tile.shape[1]
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                tm   = cv2.copyMakeBorder(tm,   0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

            cover = float((tm > 0).mean())
            if cover < MIN_MASK_COVER:
                continue

            tm01 = (tm > 0).astype(np.uint8)
            if not _tile_quality_ok(tile, tm01):
                continue

            valid_pixels, black_ratio, tex_std = _tile_quality_stats(tile, tm01)

            is_defect = 0
            defect_ratio = 0.0
            defect_pixels = 0  # <-- 항상 정의

            if defect01 is not None:
                td = defect01[y0:y0 + TILE, x0:x0 + TILE]
                if td.shape[0] != TILE or td.shape[1] != TILE:
                    pad_h = TILE - td.shape[0]
                    pad_w = TILE - td.shape[1]
                    td = cv2.copyMakeBorder(td, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

                roi_pixels = int(tm01.sum())
                if roi_pixels > 0:
                    defect_pixels = int(((td > 0) & (tm01 > 0)).sum())
                    defect_ratio = float(defect_pixels / roi_pixels)
                    is_defect = 1 if (defect_ratio >= DEFECT_RATIO_THR and defect_pixels >= MIN_DEFECT_PIXELS) else 0

            out_name = f"{stem}_x{x0}_y{y0}.png"

            # external_test이면 defect/normal 분리 저장
            if out_def_dir is not None and out_norm_dir is not None:
                out_path = (out_def_dir if is_defect else out_norm_dir) / out_name
            else:
                out_path = out_img_dir / out_name

            cv2.imwrite(str(out_path), tile)

            rows.append({
                "split": split,
                "tile": str(out_path),
                "src_stem": stem,
                "x0": x0, "y0": y0,
                "tile_size": TILE,
                "mask_cover": cover,
                "valid_pixels": valid_pixels,
                "black_ratio": black_ratio,
                "tex_std": tex_std,
                "strip_w": W, "strip_h": H,
                "is_defect": int(is_defect),
                "defect_ratio": float(defect_ratio),
                "defect_pixels": int(defect_pixels),
            })

    return rows


def _safe_crop(img, x0, y0, sz):
    H, W = img.shape[:2]
    x0 = int(np.clip(x0, 0, max(0, W - sz)))
    y0 = int(np.clip(y0, 0, max(0, H - sz)))
    return img[y0:y0+sz, x0:x0+sz], x0, y0


def _crop_cover_ok(roi01_crop):
    return float((roi01_crop > 0).mean()) >= MIN_MASK_COVER


def _save_external_crops(strip, roi01, def01, out_def_dir, out_norm_dir, debug_dir, stem):
    H, W = strip.shape[:2]
    if def01 is None:
        return 0, 0

    n, lbl, stats, cents = cv2.connectedComponentsWithStats((def01 > 0).astype(np.uint8), connectivity=8)

    # 1) 결함 crop 저장
    def_count = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_DEFECT_PIXELS_CROP:
            continue
        cx, cy = cents[i]
        x0 = int(cx - CROP_SIZE // 2)
        y0 = int(cy - CROP_SIZE // 2)

        crop, x0, y0 = _safe_crop(strip, x0, y0, CROP_SIZE)
        roi_c, _, _ = _safe_crop(roi01, x0, y0, CROP_SIZE)
        def_c, _, _ = _safe_crop(def01,  x0, y0, CROP_SIZE)

        if not _crop_cover_ok(roi_c):
            continue

        defect_pixels = int(((def_c > 0) & (roi_c > 0)).sum())
        if defect_pixels < MIN_DEFECT_PIXELS_CROP:
            continue

        # 품질 필터(ROI 기준)
        if not _tile_quality_ok(crop, (roi_c > 0).astype(np.uint8)):
            continue

        name = f"{stem}_def_{def_count:03d}_x{x0}_y{y0}.png"
        cv2.imwrite(str(out_def_dir / name), crop)
        def_count += 1

        vis = crop.copy()
        cnts, _ = cv2.findContours((def_c > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / f"{stem}_defvis_{def_count:03d}.jpg"), vis)

    # 2) 정상 crop 랜덤 샘플링
    rng = np.random.default_rng(0)
    norm_saved = 0
    tries = 0
    while norm_saved < N_NORMAL_CROPS_PER_WHEEL and tries < N_NORMAL_CROPS_PER_WHEEL * 50:
        tries += 1
        x0 = int(rng.integers(0, max(1, W - CROP_SIZE)))
        y0 = int(rng.integers(0, max(1, H - CROP_SIZE)))

        crop, x0, y0 = _safe_crop(strip, x0, y0, CROP_SIZE)
        roi_c, _, _ = _safe_crop(roi01, x0, y0, CROP_SIZE)
        def_c, _, _ = _safe_crop(def01,  x0, y0, CROP_SIZE)

        if not _crop_cover_ok(roi_c):
            continue

        defect_pixels = int(((def_c > 0) & (roi_c > 0)).sum())
        if defect_pixels > 0:
            continue

        if not _tile_quality_ok(crop, (roi_c > 0).astype(np.uint8)):
            continue

        name = f"{stem}_nor_{norm_saved:03d}_x{x0}_y{y0}.png"
        cv2.imwrite(str(out_norm_dir / name), crop)
        norm_saved += 1

    return def_count, norm_saved


def _save_calib_crops(strip, roi01, out_dir, debug_dir, stem):
    H, W = strip.shape[:2]
    rng = np.random.default_rng(0)

    saved = 0
    tries = 0
    while saved < CALIB_N_PER_IMAGE and tries < CALIB_N_PER_IMAGE * 80:
        tries += 1
        x0 = int(rng.integers(0, max(1, W - CROP_SIZE)))
        y0 = int(rng.integers(0, max(1, H - CROP_SIZE)))

        crop, x0, y0 = _safe_crop(strip, x0, y0, CROP_SIZE)
        roi_c, _, _ = _safe_crop(roi01, x0, y0, CROP_SIZE)

        if not _crop_cover_ok(roi_c):
            continue
        if not _tile_quality_ok(crop, (roi_c > 0).astype(np.uint8)):
            continue

        name = f"{stem}_cal_{saved:03d}_x{x0}_y{y0}.png"
        cv2.imwrite(str(out_dir / name), crop)
        saved += 1

    return saved


def _pred_mask_ultralytics(img_path: Path):
    from ultralytics import YOLO
    model = YOLO(str(SEG_WEIGHTS))
    res = model.predict(
        source=str(img_path),
        task="segment",
        imgsz=PRED_IMGSZ,
        conf=PRED_CONF,
        max_det=PRED_MAXDET,
        device=PRED_DEVICE,
        verbose=False,
    )[0]
    h, w = res.orig_shape
    out = np.zeros((h, w), dtype=np.uint8)
    if res.masks is None:
        return out
    for poly in res.masks.xy:
        pts = np.array(poly, dtype=np.int32)
        if pts.shape[0] >= 3:
            cv2.fillPoly(out, [pts], 1)
    return out


def _parse_mode_and_only_split():
    """
    호환 정책:
      - python preprocess.py              -> MODE_DEFAULT
      - python preprocess.py tiles        -> mode=tiles
      - python preprocess.py external_crops -> mode=external_crops
      - python preprocess.py external_test  -> (기존 호환) only_split=external_test, mode=MODE_DEFAULT
      - python preprocess.py tiles test   -> mode=tiles, only_split=test
    """
    mode = MODE_DEFAULT
    only = None

    if len(sys.argv) >= 2:
        a1 = sys.argv[1]
        if a1 in MODES:
            mode = a1
            if len(sys.argv) >= 3:
                only = sys.argv[2]
        else:
            # old style: first arg is a split
            only = a1

    return mode, only


def main():
    mode, only = _parse_mode_and_only_split()
    print(f"[MODE] {mode}   [ONLY_SPLIT] {only}")

    # splits 결정
    if mode == "tiles":
        splits = SPLITS.copy()
        if ADD_EXTERNAL:
            splits.append(EXTERNAL_SPLIT)
    elif mode in ("external_tiles", "external_crops"):
        splits = [EXTERNAL_SPLIT]
    elif mode == "calib_crops":
        splits = [CALIB_SPLIT]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if only is not None:
        splits = [only]

    all_rows = []

    for split in splits:
        img_dir = DATA_DIR / split / "images"
        lab_dir = DATA_DIR / split / "labels"

        imgs = _img_list(img_dir)
        print(f"[{split}] images={len(imgs)}  dir={img_dir}")

        dbg_split = DBG_DIR / split
        dbg_split.mkdir(parents=True, exist_ok=True)

        # output dirs
        if split == EXTERNAL_SPLIT:
            out_def = OUT_DIR / split / "defect" / "images"
            out_nor = OUT_DIR / split / "normal" / "images"
            out_def.mkdir(parents=True, exist_ok=True)
            out_nor.mkdir(parents=True, exist_ok=True)
            out_img_dir = None
        elif mode == "calib_crops":
            out_img_dir = None
            out_def = out_nor = None
            calib_dir = OUT_DIR / "calib_crops" / "normal" / "images"
            calib_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_img_dir = OUT_DIR / split / "images"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_def = out_nor = None

        for img_path in tqdm(imgs, desc=f"preprocess:{split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            label_path = lab_dir / f"{img_path.stem}.txt"
            inst = _read_yolo_instances(label_path)

            tread_polys  = [poly for cls, poly in inst if cls == TREAD_CLS]
            defect_polys = [poly for cls, poly in inst if cls == DEFECT_CLS]

            # tread ROI mask
            if USE_PRED:
                roi = _pred_mask_ultralytics(img_path)
            else:
                roi = _polys_to_mask(tread_polys, w, h)
            roi = _mask_clean_roi(roi)

            # defect mask
            defect = _polys_to_mask(defect_polys, w, h)
            defect = _mask_clean_defect(defect)

            out = _unwrap_rubber_sheet(img, roi, extra_masks=[defect])
            if out is None:
                continue
            strip, strip_roi, extras = out
            strip_def = extras[0] if extras else None

            # ---- MODE branch ----
            if split == EXTERNAL_SPLIT and mode == "external_crops":
                _save_external_crops(strip, strip_roi, strip_def, out_def, out_nor, dbg_split, img_path.stem)
                continue

            if mode == "calib_crops":
                _save_calib_crops(strip, strip_roi, calib_dir, dbg_split, img_path.stem)
                continue

            # tiles mode (in-domain + external_tiles)
            if split == EXTERNAL_SPLIT:
                rows = _tile_and_save(
                    strip, strip_roi,
                    out_img_dir=None,
                    debug_dir=dbg_split,
                    stem=img_path.stem,
                    split=split,
                    defect01=strip_def,
                    out_def_dir=out_def,
                    out_norm_dir=out_nor,
                )
            else:
                rows = _tile_and_save(
                    strip, strip_roi,
                    out_img_dir=out_img_dir,
                    debug_dir=dbg_split,
                    stem=img_path.stem,
                    split=split,
                    defect01=None,
                )
            all_rows.extend(rows)

    # calib_crops / external_crops는 tiles.csv를 만들 필요가 없음
    if mode in ("external_crops", "calib_crops"):
        print("[DONE] crops mode finished. tiles.csv not generated in this mode.")
        return

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "tiles.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)

    if len(df) == 0:
        print("No tiles generated. Check DATA_DIR/split/images path and extensions.")
    else:
        print(df.groupby("split").size())


if __name__ == "__main__":
    main()
