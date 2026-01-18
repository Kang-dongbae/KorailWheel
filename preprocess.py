# preprocess.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ROOT  # ROOT = Path(r"C:\Dev\KorailWheel") 같은 형태로 되어있다고 가정

# -------------------------
# Hyperparams (ONLY here)
# -------------------------
SPLITS = ["train", "valid", "test"]

DATA_DIR = ROOT / "data"                # 기존: data/train|valid|test/images, labels
OUT_DIR  = DATA_DIR / "data_tiles"          # 새로 생성될 타일 데이터셋
DBG_DIR  = DATA_DIR / "artifacts" / "debug_preprocess"
DBG_DIR.mkdir(parents=True, exist_ok=True)

# Unwrap (rubber-sheet) size
UNWRAP_W = 768          # 너무 크게 잡으면(예: 2048) 텍스처가 퍼져서 불리함
SMOOTH_WIN = 31         # 좌/우 경계선 스무딩(홀수 권장)
Y_MARGIN = 5            # ROI 상하 여유

# Tiling
TILE = 512
STRIDE = 256
MIN_MASK_COVER = 0.90   # 타일 안에서 유효(마스크) 비율이 이보다 작으면 버림


# 추가 필터 (타일 품질)
BLACK_THR = 10            # grayscale < 10 이면 '검정'으로 간주
MAX_BLACK_RATIO = 0.15    # ROI 내부 검정 비율이 15% 넘으면 버림
MIN_TEXTURE_STD = 12.0    # ROI 내부 grayscale 표준편차가 이 값보다 작으면(텍스처 빈약) 버림
MIN_VALID_PIXELS = 5000   # ROI 유효 픽셀이 너무 적으면 버림(안정성)


# Optional: Stage1 seg model로 마스크 예측해서 쓰고 싶으면 True
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


def _read_polys_yolo_seg(label_path: Path):
    # YOLO-seg line: cls x y w h x1 y1 x2 y2 ... (normalized polygon)
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    polys = []
    for line in txt.splitlines():
        p = line.strip().split()
        coords = np.array(list(map(float, p[5:])), dtype=np.float32)
        if coords.size >= 6:
            polys.append(coords.reshape(-1, 2))
    return polys


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


def _mask_clean(mask: np.ndarray):
    # 작은 구멍/잡음 정리
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _band_bounds(mask: np.ndarray):
    """
    mask에서 각 y마다 좌/우 경계(xL, xR)를 뽑아 rubber-sheet 언랩에 씀
    return: y0, y1, xL_arr, xR_arr (길이 = y1-y0+1)
    """
    ys = np.where(mask.sum(axis=1) > 0)[0]
    if ys.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())

    xL = []
    xR = []
    for y in range(y0, y1 + 1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            xL.append(-1)
            xR.append(-1)
        else:
            xL.append(int(xs.min()))
            xR.append(int(xs.max()))

    xL = np.array(xL, dtype=np.int32)
    xR = np.array(xR, dtype=np.int32)

    # invalid row forward/back fill
    valid = (xL >= 0) & (xR >= 0) & (xR > xL)
    if not valid.any():
        return None

    # forward fill
    last = None
    for i in range(len(xL)):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last
    # backward fill
    last = None
    for i in range(len(xL) - 1, -1, -1):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last

    # smooth boundaries
    xL = _smooth_1d(xL.astype(np.float32), SMOOTH_WIN)
    xR = _smooth_1d(xR.astype(np.float32), SMOOTH_WIN)

    return y0, y1, xL, xR


def _bounds_full(mask_slice: np.ndarray):
    """
    mask_slice(H,W) 전체 height(H)에 대해 각 row의 (xL, xR)을 만들고
    빈 row는 forward/back fill로 채움. (길이 mismatch 방지)
    """
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

    # forward fill
    last = None
    for i in range(H):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last

    # backward fill
    last = None
    for i in range(H - 1, -1, -1):
        if valid[i]:
            last = (xL[i], xR[i])
        elif last is not None:
            xL[i], xR[i] = last

    # smooth
    xL = _smooth_1d(xL.astype(np.float32), SMOOTH_WIN)
    xR = _smooth_1d(xR.astype(np.float32), SMOOTH_WIN)

    return xL, xR


def _unwrap_rubber_sheet(img: np.ndarray, mask: np.ndarray):
    h, w = img.shape[:2]

    bb = _band_bounds(mask)  # 여기서는 y0/y1만 쓰려고 호출
    if bb is None:
        return None
    y0_old, y1_old, _, _ = bb

    y0 = max(0, y0_old - Y_MARGIN)
    y1 = min(h - 1, y1_old + Y_MARGIN)

    mask_slice = mask[y0:y1 + 1, :]
    H = mask_slice.shape[0]
    ys = np.arange(y0, y1 + 1, dtype=np.float32)

    bounds = _bounds_full(mask_slice)
    if bounds is None:
        return None
    xL2, xR2 = bounds

    map_x = np.zeros((H, UNWRAP_W), dtype=np.float32)
    map_y = np.zeros((H, UNWRAP_W), dtype=np.float32)

    for i in range(H):
        xl = float(xL2[i])
        xr = float(xR2[i])
        if xr <= xl + 1:
            xr = xl + 1.0

        xs = np.linspace(xl, xr, UNWRAP_W, dtype=np.float32)
        map_x[i, :] = np.clip(xs, 0, w - 1)
        map_y[i, :] = ys[i]  # 중요: 픽셀 y좌표 그대로

    strip = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    m = (mask.astype(np.float32) * 255.0)
    strip_m = cv2.remap(
        m, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    strip_m = (strip_m > 0).astype(np.uint8)

    return strip, strip_m

def _tile_quality_stats(tile_bgr: np.ndarray, tile_mask01: np.ndarray):
    # ROI(mask==1) 내부에서만 통계 계산
    valid = (tile_mask01 > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0, 1.0, 0.0  # valid_pixels, black_ratio, tex_std

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



def _tile_and_save(strip: np.ndarray, strip_m: np.ndarray, out_img_dir: Path, debug_dir: Path, stem: str, split: str):
    H, W = strip.shape[:2]
    rows = []

    # 디버그로 base 저장
    cv2.imwrite(str(debug_dir / f"{stem}_base.png"), strip)

    # 슬라이딩 타일
    for y0 in range(0, max(1, H - TILE + 1), STRIDE):
        for x0 in range(0, max(1, W - TILE + 1), STRIDE):
            tile = strip[y0:y0 + TILE, x0:x0 + TILE]
            tm   = strip_m[y0:y0 + TILE, x0:x0 + TILE]

            # pad if needed (끝자락)
            if tile.shape[0] != TILE or tile.shape[1] != TILE:
                pad_h = TILE - tile.shape[0]
                pad_w = TILE - tile.shape[1]
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                tm   = cv2.copyMakeBorder(tm,   0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

            cover = float((tm > 0).mean())
            if cover < MIN_MASK_COVER:
                continue

            # ✅ 0/1 마스크로 보장
            tm01 = (tm > 0).astype(np.uint8)

            # ✅ 타일 품질 필터(검정/텍스처)
            if not _tile_quality_ok(tile, tm01):
                continue

            valid_pixels, black_ratio, tex_std = _tile_quality_stats(tile, tm01)

            out_name = f"{stem}_x{x0}_y{y0}.png"
            out_path = out_img_dir / out_name
            cv2.imwrite(str(out_path), tile)

            rows.append({
                "split": split,
                "tile": str(out_path),
                "src_stem": stem,
                "x0": x0,
                "y0": y0,
                "tile_size": TILE,
                "mask_cover": cover,
                "valid_pixels": valid_pixels,
                "black_ratio": black_ratio,
                "tex_std": tex_std,
                "strip_w": W,
                "strip_h": H,
            })

    return rows


# -------------------------
# Optional: predict mask (for unlabeled images later)
# -------------------------
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
    # xy는 orig 좌표 폴리곤 리스트라 안전(이전 broadcast 에러 방지)
    for poly in res.masks.xy:
        pts = np.array(poly, dtype=np.int32)
        if pts.shape[0] >= 3:
            cv2.fillPoly(out, [pts], 1)
    return out


def main():
    all_rows = []

    for split in SPLITS:
        img_dir = DATA_DIR / split / "images"
        lab_dir = DATA_DIR / split / "labels"

        out_img_dir = OUT_DIR / split / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        dbg_split = DBG_DIR / split
        dbg_split.mkdir(parents=True, exist_ok=True)

        imgs = _img_list(img_dir)
        for img_path in tqdm(imgs, desc=f"preprocess:{split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            if USE_PRED:
                mask = _pred_mask_ultralytics(img_path)
            else:
                label_path = lab_dir / f"{img_path.stem}.txt"
                polys = _read_polys_yolo_seg(label_path)
                mask = _polys_to_mask(polys, w, h)

            mask = _mask_clean(mask)

            out = _unwrap_rubber_sheet(img, mask)
            if out is None:
                continue
            strip, strip_m = out

            rows = _tile_and_save(strip, strip_m, out_img_dir, dbg_split, img_path.stem, split)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "tiles.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
    print(df.groupby("split").size())


if __name__ == "__main__":
    main()
