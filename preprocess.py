# preprocess.py
"""
사용법
- 전체(기본): python preprocess.py
- tiles 모드 명시: python preprocess.py tiles
- external만: python preprocess.py external_tiles
- 특정 split만(호환): python preprocess.py train  / valid / test / external_test
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ROOT

# =========================================================
# INPUT/OUTPUT PATHS
# =========================================================
# 입력 데이터셋 (요청하신 경로 구조)
#   ROOT/data/st1_roi/{train,valid,test,external_test}/{images,labels}
DATA_DIR = ROOT / "data" / "st1_roi"

# 출력 타일 (기존 학습 코드 호환 유지)
#   ROOT/data/data_tiles/{train,valid,test}/images
#   ROOT/data/data_tiles/external_test/{normal,defect}/images
OUT_DIR = ROOT / "data" / "data_tiles"

DBG_DIR = ROOT / "data" / "artifacts" / "debug_preprocess"
DBG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# SPLIT / MODE
# =========================================================
SPLITS = ["train", "valid", "test"]
ADD_EXTERNAL = True
EXTERNAL_SPLIT = "external_test"

MODE_DEFAULT = "tiles"
MODES = {"tiles", "external_tiles"}

# =========================================================
# UNWRAP (정규화) HYPERPARAMS
# =========================================================
UNWRAP_W = 768
SMOOTH_WIN = 31
Y_MARGIN = 5

# =========================================================
# TILING HYPERPARAMS (in-domain)
# =========================================================
TILE = 512
STRIDE = 256
MIN_MASK_COVER = 0.90

# 품질 필터(in-domain) - 유지
BLACK_THR = 10
MAX_BLACK_RATIO = 0.15
MIN_TEXTURE_STD = 12.0
MIN_VALID_PIXELS = 5000

# =========================================================
# TILING HYPERPARAMS (external 전용 설계)
# =========================================================
STRIDE_EXT = 128            # 외부는 더 촘촘히(커버리지↑)
MIN_MASK_COVER_EXT = 0.85   # 외부 ROI 흔들림 대비 완화

# 외부 "정상 타일" 필터(너무 쓰레기만 제거) - 완화
MAX_BLACK_RATIO_EXT = 0.40
MIN_TEXTURE_STD_EXT = 6.0
MIN_VALID_PIXELS_EXT = 3000

# 외부 "결함 타일" 라벨링(미검출 방지 목적이면 작게)
#   - GT 결함 픽셀 1개라도 있으면 defect로 잡고 싶으면 1
MIN_DEFECT_PIXELS_EXT = 1

# =========================================================
# LABEL / CLASS
# =========================================================
TREAD_CLS = 0
DEFECT_CLS = 1

# Optional: ROI를 seg model로 예측해서 쓰고 싶으면 True
USE_PRED = False
SEG_WEIGHTS = ROOT / "runs" / "tread_seg_stage1" / "weights" / "best.pt"
PRED_IMGSZ = 1024
PRED_CONF = 0.25
PRED_MAXDET = 5
PRED_DEVICE = "0"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# UTILS
# =========================================================
def _img_list(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

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

def _tile_quality_ok(tile_bgr, tile_mask01, *, max_black_ratio, min_texture_std, min_valid_pixels):
    valid_pixels, black_ratio, tex_std = _tile_quality_stats(tile_bgr, tile_mask01)
    if valid_pixels < min_valid_pixels:
        return False
    if black_ratio > max_black_ratio:
        return False
    if tex_std < min_texture_std:
        return False
    return True

def _sliding_starts(length: int, tile: int, stride: int):
    """끝부분 누락 방지: 마지막 시작점(length-tile)을 강제 포함."""
    if length <= tile:
        return [0]
    last = length - tile
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))

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

def _tile_and_save(
    strip, roi01,
    *,
    stem: str, split: str, debug_dir: Path, out_img_dir: Path | None,
    defect01=None, out_def_dir: Path | None = None, out_norm_dir: Path | None = None,
    stride: int = STRIDE, min_mask_cover: float = MIN_MASK_COVER, is_external: bool = False,
):
    H, W = strip.shape[:2]
    rows = []
    ys = _sliding_starts(H, TILE, stride)
    xs = _sliding_starts(W, TILE, stride)

    for y0 in ys:
        for x0 in xs:
            tile = strip[y0:y0 + TILE, x0:x0 + TILE]
            tm   = roi01[y0:y0 + TILE, x0:x0 + TILE]

            # pad
            if tile.shape[0] != TILE or tile.shape[1] != TILE:
                pad_h = TILE - tile.shape[0]; pad_w = TILE - tile.shape[1]
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                tm   = cv2.copyMakeBorder(tm,   0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

            cover = float((tm > 0).mean())
            if cover < min_mask_cover: continue
            tm01 = (tm > 0).astype(np.uint8)

            is_defect = 0
            defect_pixels = 0
            defect_ratio = 0.0
            
            td_01 = None  # <--- ★ [수정됨] 변수 초기화 추가 ★

            if defect01 is not None:
                td = defect01[y0:y0 + TILE, x0:x0 + TILE]
                if td.shape[0] != TILE or td.shape[1] != TILE:
                    pad_h = TILE - td.shape[0]; pad_w = TILE - td.shape[1]
                    td = cv2.copyMakeBorder(td, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                
                roi_pixels = int(tm01.sum())
                if roi_pixels > 0:
                    td_01 = (td > 0).astype(np.uint8)
                    defect_pixels = int((td_01 & tm01).sum())
                    defect_ratio = float(defect_pixels / roi_pixels)
                    if is_external:
                        is_defect = 1 if defect_pixels >= MIN_DEFECT_PIXELS_EXT else 0

            # 품질 필터
            if not is_external:
                if not _tile_quality_ok(tile, tm01, max_black_ratio=MAX_BLACK_RATIO, min_texture_std=MIN_TEXTURE_STD, min_valid_pixels=MIN_VALID_PIXELS):
                    continue
            else:
                if is_defect == 0:
                    if not _tile_quality_ok(tile, tm01, max_black_ratio=MAX_BLACK_RATIO_EXT, min_texture_std=MIN_TEXTURE_STD_EXT, min_valid_pixels=MIN_VALID_PIXELS_EXT):
                        continue
                else:
                    valid_pixels, black_ratio, _ = _tile_quality_stats(tile, tm01)
                    if valid_pixels < 1000 or black_ratio > 0.85: continue

            out_name = f"{stem}_x{x0}_y{y0}.png"

            # 경로 결정
            if is_external and (out_def_dir is not None) and (out_norm_dir is not None):
                base_dir = out_def_dir if is_defect else out_norm_dir
            else:
                base_dir = out_img_dir
            
            out_path = base_dir / out_name
            cv2.imwrite(str(out_path), tile)

            # 라벨(.txt) 생성 및 저장
            if is_external and is_defect and (td_01 is not None):
                out_lbl_dir = base_dir.parent / "labels"
                out_lbl_dir.mkdir(parents=True, exist_ok=True)
                
                out_lbl_path = out_lbl_dir / f"{out_path.stem}.txt"
                
                lines = _mask_to_yolo_lines(td_01, 0)
                if lines:
                    out_lbl_path.write_text("\n".join(lines), encoding="utf-8")

            valid_pixels, black_ratio, tex_std = _tile_quality_stats(tile, tm01)
            rows.append({
                "split": split,
                "tile": str(out_path),
                "is_defect": int(is_defect),
                "defect_pixels": int(defect_pixels),
                "valid_pixels": valid_pixels,
                "black_ratio": black_ratio,
                "tex_std": tex_std,
            })

    return rows

def _parse_mode_and_only_split():
    mode = MODE_DEFAULT
    only = None
    if len(sys.argv) >= 2:
        a1 = sys.argv[1]
        if a1 in MODES:
            mode = a1
            if len(sys.argv) >= 3:
                only = sys.argv[2]
        else:
            # old style: first arg is split name
            only = a1
    return mode, only

def _mask_to_yolo_lines(mask01: np.ndarray, cls_id: int):
    # 마스크에서 컨투어(테두리) 추출
    cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < 5: continue # 너무 작은 점은 무시
        
        # 다각형 단순화 (점 개수 줄이기)
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        # 정규화 (0~1 사이 값으로 변환)
        points = approx.reshape(-1, 2).astype(np.float32)
        points[:, 0] /= mask01.shape[1] # w
        points[:, 1] /= mask01.shape[0] # h
        
        # YOLO 포맷 문자열 생성
        coord_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
        lines.append(f"{cls_id} {coord_str}")
    return lines

def main():
    mode, only = _parse_mode_and_only_split()
    print(f"[MODE] {mode}   [ONLY_SPLIT] {only}")

    # splits 결정
    if mode == "tiles":
        splits = SPLITS.copy()
        if ADD_EXTERNAL:
            splits.append(EXTERNAL_SPLIT)
    elif mode == "external_tiles":
        splits = [EXTERNAL_SPLIT]
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
            (OUT_DIR / split / "defect" / "labels").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / split / "normal" / "labels").mkdir(parents=True, exist_ok=True)
            out_img_dir = None
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

            # external 설계 반영: stride/cover/필터 정책만 다르게
            if split == EXTERNAL_SPLIT:
                rows = _tile_and_save(
                    strip, strip_roi,
                    stem=img_path.stem,
                    split=split,
                    debug_dir=dbg_split,
                    out_img_dir=None,
                    defect01=strip_def,
                    out_def_dir=out_def,
                    out_norm_dir=out_nor,
                    stride=STRIDE_EXT,
                    min_mask_cover=MIN_MASK_COVER_EXT,
                    is_external=True,
                )
            else:
                rows = _tile_and_save(
                    strip, strip_roi,
                    stem=img_path.stem,
                    split=split,
                    debug_dir=dbg_split,
                    out_img_dir=out_img_dir,
                    defect01=None,
                    stride=STRIDE,
                    min_mask_cover=MIN_MASK_COVER,
                    is_external=False,
                )

            all_rows.extend(rows)

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
