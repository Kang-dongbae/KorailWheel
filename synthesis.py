# synth_patch_to_tiles.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ROOT

# -------------------------
# Paths
# -------------------------
TILES_DIR = ROOT / "data" / "data_tiles"            # train/valid/test/images
PATCH_DIR = ROOT / "data" / "patch"                 # train/valid/test/images, labels
OUT_DIR   = ROOT / "data" / "data_tiles_synth"      # output yolo-seg dataset

# -------------------------
# Synthesis hyperparams
# -------------------------
SEED = 0

# output includes: original normals + (optional) synthetic defect versions
COPY_NORMALS = True

# probability to create a synthetic-defect version per tile (per split)
P_DEFECT_TRAIN = 0.45
P_DEFECT_VALID = 0.25
P_DEFECT_TEST  = 0.25

# max patches pasted per synthetic tile
MAX_PATCHES_PER_TILE = 1

# target defect area ratio within 512x512 tile (mask area / tile area)
# external 실결함이 "작은 박리/치핑"이면 너무 크게 붙이지 않는게 중요
AREA_RATIO_MIN = 0.003   # 0.3%
AREA_RATIO_MAX = 0.02    # 6%

# scale clamp (avoid huge upscaling artifacts)
SCALE_MIN = 0.6
SCALE_MAX = 1.4

# placement validity (avoid black/empty regions)
VALID_GRAY_THR = 12
MIN_VALID_OVERLAP = 0.90  # mask pixels overlap with valid(gray>thr) pixels

# mask feather (edge blend)
FEATHER_K = 11   # odd, 7~15 추천
FEATHER_SIGMA = 0

# photometric match strength (0~1)
PHOTO_MATCH = True

# small post effects to hide seams
POST_JPEG_QUALITY = 92  # e.g. 92 if you want re-jpeg; None to disable
POST_GAUSS_BLUR = 0       # e.g. 1 or 2 (kernel= (2*blur+1))
POST_NOISE_STD = 0.0      # e.g. 2.0

# Patch label class id in PATCH_DIR labels (보통 0 하나일 확률 큼)
PATCH_CLS = 0
# Output defect class id in synthesized dataset labels
OUT_DEFECT_CLS = 0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

USE_POISSON = True
POISSON_MODE = cv2.NORMAL_CLONE   # 또는 cv2.MIXED_CLONE
FEATHER_PX = 10   # 경계가 사라지는 폭(픽셀). 8~16 추천

USE_SEAMLESS_CLONE = True
SEAMLESS_MODE = cv2.NORMAL_CLONE   # 또는 cv2.MIXED_CLONE
MASK_ERODE = 1                     # 경계 1~2픽셀 줄여서 테두리 제거



# -------------------------
# Utils: list images
# -------------------------
def _img_list(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


# -------------------------
# Utils: read YOLO seg instances (bbox-only or seg)
# -------------------------
def _read_yolo_instances(label_path: Path):
    """
    Accepts:
      - seg (roboflow style): cls x y w h x1 y1 x2 y2 ...
      - bbox only: cls x y w h
    returns list of (cls:int, poly_norm:(N,2) float32)
    """
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    out = []
    for line in txt.splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))

        if len(p) == 5:
            xc, yc, bw, bh = map(float, p[1:5])
            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2
            poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
            out.append((cls, poly))
        else:
            coords = np.array(list(map(float, p[5:])), dtype=np.float32)
            if coords.size >= 6:
                out.append((cls, coords.reshape(-1, 2)))
    return out


def _poly_to_mask(poly_norm: np.ndarray, w: int, h: int):
    m = np.zeros((h, w), dtype=np.uint8)
    pts = poly_norm.copy()
    pts[:, 0] = np.clip(pts[:, 0] * w, 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1] * h, 0, h - 1)
    pts = pts.astype(np.int32)
    if pts.shape[0] >= 3:
        cv2.fillPoly(m, [pts], 255)
    return m


def _extract_patch_instances(split: str):
    """
    Build in-memory list of patch instances: each item has (rgb, mask255)
    """
    img_dir = PATCH_DIR / split / "images"
    lab_dir = PATCH_DIR / split / "labels"

    items = []
    for img_path in _img_list(img_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        inst = _read_yolo_instances(lab_dir / f"{img_path.stem}.txt")

        # keep only PATCH_CLS
        polys = [poly for cls, poly in inst if cls == PATCH_CLS]
        if not polys:
            continue

        for i, poly in enumerate(polys):
            mask = _poly_to_mask(poly, w, h)
            if mask.sum() < 200:   # too tiny
                continue
            x, y, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8))
            # add margin (avoid too tight crops -> seam)
            margin = int(0.10 * max(bw, bh)) + 3
            x0 = max(0, x - margin); y0 = max(0, y - margin)
            x1 = min(w, x + bw + margin); y1 = min(h, y + bh + margin)

            crop = img[y0:y1, x0:x1].copy()
            m    = mask[y0:y1, x0:x1].copy()

            # ensure mask is 0/255
            m = (m > 0).astype(np.uint8) * 255

            # --- PATCH QC (ADD HERE) ---
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            H,S,V = cv2.split(hsv)

            mask_in = (m > 0)
            if mask_in.sum() < 200:
                continue

            S_mean = float(S[mask_in].mean())
            V_mean = float(V[mask_in].mean())

            # 채도/밝기 과도한 패치 컷 (값은 데이터 보면서 미세조정)
            if S_mean > 80:      # 색이 너무 화려함(청록/무지개 원인 방지)
                continue
            if V_mean > 245 or V_mean < 10:   # 과다노출/암부 패치 컷
                continue

            items.append({
                "src": str(img_path),
                "inst_idx": i,
                "rgb": crop,
                "mask": m
            })

    return items


def _feather_alpha(mask255: np.ndarray):
    m = (mask255 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(np.float32)

    # inside distance (경계에서 안쪽으로 거리)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)

    # alpha: 경계 부근은 0~1로 천천히 올라가고, 내부는 1
    a = np.clip(dist_in / float(FEATHER_PX), 0.0, 1.0).astype(np.float32)

    # 아주 약하게만 스무딩(선택)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    return a


def _photometric_match_lab(patch_bgr: np.ndarray, tile_bgr: np.ndarray, alpha: np.ndarray):
    m = (alpha > 0.3)
    if m.sum() < 50:
        return patch_bgr

    p_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(tile_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)

    out = p_lab.copy()
    for c in range(3):
        pv = p_lab[..., c][m]
        tv = t_lab[..., c][m]
        p_mean, p_std = float(pv.mean()), float(pv.std() + 1e-6)
        t_mean, t_std = float(tv.mean()), float(tv.std() + 1e-6)
        out[..., c] = (out[..., c] - p_mean) * (t_std / p_std) + t_mean

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _photometric_match(patch_bgr: np.ndarray, tile_bgr: np.ndarray, alpha: np.ndarray):
    # match mean/std inside alpha>0.3 region
    m = (alpha > 0.3)
    if m.sum() < 50:
        return patch_bgr

    p = patch_bgr.astype(np.float32)
    t = tile_bgr.astype(np.float32)

    out = p.copy()
    for c in range(3):
        pv = p[..., c][m]
        tv = t[..., c][m]
        p_mean, p_std = float(pv.mean()), float(pv.std() + 1e-6)
        t_mean, t_std = float(tv.mean()), float(tv.std() + 1e-6)
        out[..., c] = (out[..., c] - p_mean) * (t_std / p_std) + t_mean

    return np.clip(out, 0, 255).astype(np.uint8)


def _random_transform_patch(rng, rgb, mask255, tile_size=512):
    # decide target area ratio
    ar = float(rng.uniform(AREA_RATIO_MIN, AREA_RATIO_MAX))
    tile_area = tile_size * tile_size
    mask_area = int((mask255 > 0).sum())
    if mask_area <= 0:
        return None

    s = (ar * tile_area / mask_area) ** 0.5
    s = float(np.clip(s, SCALE_MIN, SCALE_MAX))

    # scale
    ph, pw = rgb.shape[:2]
    nh = max(8, int(ph * s))
    nw = max(8, int(pw * s))
    rgb2  = cv2.resize(rgb,  (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask2 = cv2.resize(mask255, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # rotation (small, to avoid unrealistic orientation)
    ang = float(rng.uniform(-5, 5))
    M = cv2.getRotationMatrix2D((nw/2, nh/2), ang, 1.0)
    rgb3  = cv2.warpAffine(
        rgb2, M, (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    mask3 = cv2.warpAffine(
        mask2, M, (nw, nh),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # occasional flip
    if rng.random() < 0.15:
        rgb3 = cv2.flip(rgb3, 1)
        mask3 = cv2.flip(mask3, 1)

    # keep only if still valid
    if (mask3 > 0).sum() < 200:
        return None

    return rgb3, mask3


def _find_valid_position(rng, tile_bgr, alpha, max_tries=80):
    H, W = tile_bgr.shape[:2]
    ph, pw = alpha.shape[:2]

    if ph >= H or pw >= W:
        return None

    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    valid = (g > VALID_GRAY_THR).astype(np.uint8)

    am = (alpha > 0.3).astype(np.uint8)

    for _ in range(max_tries):
        x0 = int(rng.integers(0, W - pw))
        y0 = int(rng.integers(0, H - ph))
        v = valid[y0:y0+ph, x0:x0+pw]
        overlap = float(v[am > 0].mean()) if (am > 0).any() else 0.0
        if overlap >= MIN_VALID_OVERLAP:
            return x0, y0
    return None


def _mask_to_yolo_seg_lines(mask01: np.ndarray, cls_id: int):
    H, W = mask01.shape[:2]
    cnts, _ = cv2.findContours((mask01 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 30:  # too tiny
            continue

        # simplify polygon
        eps = 0.002 * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True)
        if ap.shape[0] < 3:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        xc = (x + bw/2) / W
        yc = (y + bh/2) / H
        ww = bw / W
        hh = bh / H

        pts = ap.reshape(-1, 2).astype(np.float32)
        coords = []
        for px, py in pts:
            coords.append(f"{px / W:.6f}")
            coords.append(f"{py / H:.6f}")

        line = f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f} " + " ".join(coords)
        lines.append(line)

    return lines


def _post_effects(rng, img_bgr):
    out = img_bgr

    if POST_GAUSS_BLUR > 0:
        k = POST_GAUSS_BLUR * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    if POST_NOISE_STD and POST_NOISE_STD > 0:
        noise = rng.normal(0, POST_NOISE_STD, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if POST_JPEG_QUALITY is not None:
        q = int(POST_JPEG_QUALITY)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def _p_defect_for_split(split: str):
    if split == "train":
        return P_DEFECT_TRAIN
    if split == "valid":
        return P_DEFECT_VALID
    return P_DEFECT_TEST


def synth_split(split: str, patch_items):
    rng = np.random.default_rng(SEED + {"train": 0, "valid": 1, "test": 2}.get(split, 9))

    in_img_dir = TILES_DIR / split / "images"
    out_img_dir = OUT_DIR / split / "images"
    out_lab_dir = OUT_DIR / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    imgs = _img_list(in_img_dir)
    p_def = _p_defect_for_split(split)

    meta_rows = []

    for img_path in tqdm(imgs, desc=f"synth:{split}"):
        tile = cv2.imread(str(img_path))
        if tile is None:
            continue

        # 1) copy original as normal
        if COPY_NORMALS:
            out_name = img_path.name
            cv2.imwrite(str(out_img_dir / out_name), tile)
            (out_lab_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")

        # 2) create synthetic-defect version (optional)
        if rng.random() > p_def:
            continue
        if not patch_items:
            continue

        syn = tile.copy()
        H, W = syn.shape[:2]
        full_mask = np.zeros((H, W), dtype=np.uint8)

        n_patches = int(rng.integers(1, MAX_PATCHES_PER_TILE + 1))
        used = 0

        for _ in range(n_patches):
            item = patch_items[int(rng.integers(0, len(patch_items)))]
            tr = _random_transform_patch(rng, item["rgb"], item["mask"], tile_size=W)
            if tr is None:
                continue
            prgb, pmask = tr
            alpha = _feather_alpha(pmask)

            pos = _find_valid_position(rng, syn, alpha)
            if pos is None:
                continue
            x0, y0 = pos
            ph, pw = prgb.shape[:2]

            roi = syn[y0:y0+ph, x0:x0+pw].copy()

            # photometric match on-the-fly
            if PHOTO_MATCH:
                prgb = _photometric_match_lab(prgb, roi, alpha)

            # blend
            # mask erode로 테두리/잡티 제거
            pmask2 = pmask.copy()
            if MASK_ERODE > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                pmask2 = cv2.erode(pmask2, k, iterations=MASK_ERODE)

            m01 = (pmask2 > 0).astype(np.uint8)

            if USE_SEAMLESS_CLONE:
                src_canvas = np.zeros_like(syn)
                m_canvas   = np.zeros((H, W), dtype=np.uint8)

                src_canvas[y0:y0+ph, x0:x0+pw] = prgb
                m_canvas[y0:y0+ph, x0:x0+pw]   = (m01 * 255)

                center = (x0 + pw // 2, y0 + ph // 2)
                syn = cv2.seamlessClone(src_canvas, syn, m_canvas, center, SEAMLESS_MODE)
            else:
                alpha2 = _feather_alpha(m01 * 255)
                a3 = alpha2[..., None].astype(np.float32)
                blended = (prgb.astype(np.float32) * a3 + roi.astype(np.float32) * (1.0 - a3)).astype(np.uint8)
                syn[y0:y0+ph, x0:x0+pw] = blended

            # update mask (GT)
            full_mask[y0:y0+ph, x0:x0+pw] = np.maximum(full_mask[y0:y0+ph, x0:x0+pw], m01)

            # update mask (binary)
            m01 = (pmask > 0).astype(np.uint8)
            full_mask[y0:y0+ph, x0:x0+pw] = np.maximum(full_mask[y0:y0+ph, x0:x0+pw], m01)

            used += 1
            meta_rows.append({
                "split": split,
                "tile": str(img_path),
                "patch_src": item["src"],
                "patch_inst": int(item["inst_idx"]),
                "x0": x0, "y0": y0,
                "ph": ph, "pw": pw
            })

        if used == 0:
            continue

        syn = _post_effects(rng, syn)

        out_name = f"{img_path.stem}__syn.png"
        cv2.imwrite(str(out_img_dir / out_name), syn)

        # label file (YOLOv8-seg style with bbox + polygon)
        lines = _mask_to_yolo_seg_lines(full_mask, OUT_DEFECT_CLS)
        (out_lab_dir / f"{Path(out_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    # meta
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(OUT_DIR / f"meta_{split}.csv", index=False, encoding="utf-8-sig")


def main():
    for split in ["train", "valid", "test"]:
        print(f"\n[1/2] load patches: {split}")
        patch_items = _extract_patch_instances(split)
        print(f"  patch instances: {len(patch_items)}")

        print(f"[2/2] synth tiles: {split}")
        synth_split(split, patch_items)

    print("\nDONE.")
    print("Synth dataset:", OUT_DIR)


if __name__ == "__main__":
    main()
