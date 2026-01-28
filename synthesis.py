# synth_patch_to_tiles.py  (FINAL OPT)
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ROOT

# -------------------------
# Paths
# -------------------------
TILES_DIR = ROOT / "data" / "data_tiles"
PATCH_DIR = ROOT / "data" / "patch"
OUT_DIR   = ROOT / "data" / "data_tiles_synth"

# -------------------------
# Synthesis hyperparams
# -------------------------
SEED = 0
COPY_NORMALS = True

P_DEFECT_TRAIN = 0.35
P_DEFECT_VALID = 0.20
P_DEFECT_TEST  = 0.20

MAX_PATCHES_PER_TILE = 2

# [OPT] 외부 실사(박리/치핑 군집) 커버 + log-uniform 유지
AREA_RATIO_MIN = 0.0007   # 0.05%
AREA_RATIO_MAX = 0.015   # 4.0%

SCALE_MIN = 0.25
SCALE_MAX = 1.00

VALID_GRAY_THR = 12
MIN_VALID_OVERLAP = 0.85  # [OPT] 0.95 -> 완화

# feather
FEATHER_PX = 10           # [OPT] 경계 부드럽게 (6 -> 10)
MASK_ERODE = 1

# photometric match
PHOTO_MATCH = True
PHOTO_L_ONLY = True
PHOTO_STRENGTH = 0.30    # [OPT] 결함 특징 죽이지 않게 약하게

# [OPT] 결함 강도: 과도한 대비를 줄여 FP(얼룩) 억제
DEFECT_STRENGTH_MIN = 0.90
DEFECT_STRENGTH_MAX = 1.15

# seamlessClone (기본 OFF 유지)
USE_SEAMLESS_CLONE = True
SEAMLESS_MODE = cv2.NORMAL_CLONE
USE_SEAMLESS_PROB = 0.0
DE_SEAMLESS_MAX = 8.0

# [OPT] gradient 강제 제거(밴딩 과민 방지): “선호”만 하고 강제하지 않음
GRAD_MEAN_THR = 2.0       # 참고값
P_ACCEPT_LOW_GRAD = 0.55  # [OPT] grad 낮아도 55%는 허용

# ---- Hard-negative (stain) normals ----
# [OPT] 정상에서 밴딩/얼룩 다양성을 더 많이 보여줘서 FP 줄임
P_STAIN_TRAIN = 0.25
P_STAIN_VALID = 0.15
P_STAIN_TEST  = 0.15
STAIN_MAX_BLOBS = 2
STAIN_ALPHA_MIN = 0.05
STAIN_ALPHA_MAX = 0.20
STAIN_L_SHIFT = 10.0
STAIN_AB_SHIFT = 4.0

# 전역 스타일 증강은 일단 OFF(합성/실사 갭 커짐 방지)
GLOBAL_STYLE_AUG = False
GAMMA_RANGE = (0.90, 1.15)
CONTRAST_RANGE = (0.90, 1.12)
BRIGHT_BETA = (-10, 10)
AUG_BLUR_PROB = 0.10
AUG_NOISE_STD = (0.0, 1.0)
AUG_JPEG_QUALITY = (88, 98)

POST_JPEG_QUALITY = None
POST_GAUSS_BLUR = 0
POST_NOISE_STD = 0.0

PATCH_CLS = 0
OUT_DEFECT_CLS = 0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# [OPT] 라벨 품질: 작은 조각 너무 작은 건 버림
MIN_COMPONENT_AREA = 25   # pixels
MAX_POLY_POINTS = 180     # polygon 단순화 상한


# -------------------------
# Utils
# -------------------------
def _img_list(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def _read_yolo_instances(label_path: Path):
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
    img_dir = PATCH_DIR / split / "images"
    lab_dir = PATCH_DIR / split / "labels"

    items = []
    for img_path in _img_list(img_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        inst = _read_yolo_instances(lab_dir / f"{img_path.stem}.txt")

        polys = [poly for cls, poly in inst if cls == PATCH_CLS]
        if not polys:
            continue

        for i, poly in enumerate(polys):
            mask = _poly_to_mask(poly, w, h)
            if mask.sum() < 200:
                continue
            x, y, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8))
            margin = int(0.10 * max(bw, bh)) + 3
            x0 = max(0, x - margin); y0 = max(0, y - margin)
            x1 = min(w, x + bw + margin); y1 = min(h, y + bh + margin)

            crop = img[y0:y1, x0:x1].copy()
            m    = mask[y0:y1, x0:x1].copy()
            m = (m > 0).astype(np.uint8) * 255

            # Patch QC (색 과도/과노출/암부 컷)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            H,S,V = cv2.split(hsv)
            mask_in = (m > 0)
            if mask_in.sum() < 200:
                continue
            S_mean = float(S[mask_in].mean())
            V_mean = float(V[mask_in].mean())
            if S_mean > 80:
                continue
            if V_mean > 245 or V_mean < 10:
                continue

            items.append({"src": str(img_path), "inst_idx": i, "rgb": crop, "mask": m})
    return items


def _feather_alpha(mask255: np.ndarray):
    m = (mask255 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(np.float32)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    a = np.clip(dist_in / float(FEATHER_PX), 0.0, 1.0).astype(np.float32)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    return a


def _photometric_match_lab(patch_bgr: np.ndarray, tile_bgr: np.ndarray, alpha: np.ndarray):
    m = (alpha > 0.3)
    if m.sum() < 50:
        return patch_bgr

    p_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(tile_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)

    out = p_lab.copy()
    chs = [0] if PHOTO_L_ONLY else [0,1,2]
    for c in chs:
        pv = p_lab[..., c][m]
        tv = t_lab[..., c][m]
        p_mean, p_std = float(pv.mean()), float(pv.std() + 1e-6)
        t_mean, t_std = float(tv.mean()), float(tv.std() + 1e-6)
        out[..., c] = (out[..., c] - p_mean) * (t_std / p_std) + t_mean

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _deltaE_mean(patch_bgr: np.ndarray, roi_bgr: np.ndarray, mask01: np.ndarray) -> float:
    m = (mask01 > 0)
    if m.sum() < 50:
        return 0.0
    p = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(roi_bgr,   cv2.COLOR_BGR2LAB).astype(np.float32)
    d = p - r
    de = np.sqrt(d[...,0]**2 + d[...,1]**2 + d[...,2]**2)
    return float(de[m].mean())


def _lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _match_sharpness(patch_bgr: np.ndarray, roi_bgr: np.ndarray, alpha: np.ndarray):
    m = (alpha > 0.3)
    if m.sum() < 50:
        return patch_bgr
    pg = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    pv = _lap_var(pg)
    rv = _lap_var(rg)
    if pv > rv * 1.25:
        ratio = pv / (rv + 1e-6)
        sigma = float(np.clip(0.6 * np.log(ratio), 0.4, 1.6))
        k = int(2 * round(3 * sigma) + 1)
        patch_bgr = cv2.GaussianBlur(patch_bgr, (k, k), sigma)
    return patch_bgr


def _random_transform_patch(rng, rgb, mask255, tile_size=512):
    ar = float(np.exp(rng.uniform(np.log(AREA_RATIO_MIN), np.log(AREA_RATIO_MAX))))
    tile_area = tile_size * tile_size
    mask_area = int((mask255 > 0).sum())
    if mask_area <= 0:
        return None

    s = (ar * tile_area / mask_area) ** 0.5
    s = float(np.clip(s, SCALE_MIN, SCALE_MAX))

    ph, pw = rgb.shape[:2]
    nh = max(8, int(ph * s))
    nw = max(8, int(pw * s))
    rgb2  = cv2.resize(rgb,  (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask2 = cv2.resize(mask255, (nw, nh), interpolation=cv2.INTER_NEAREST)

    ang = float(rng.uniform(-5, 5))
    M = cv2.getRotationMatrix2D((nw/2, nh/2), ang, 1.0)
    rgb3  = cv2.warpAffine(rgb2, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    mask3 = cv2.warpAffine(mask2, M, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if rng.random() < 0.15:
        rgb3 = cv2.flip(rgb3, 1)
        mask3 = cv2.flip(mask3, 1)

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
        if overlap < MIN_VALID_OVERLAP:
            continue

        # [OPT] grad는 "선호"만 하고 강제하지 않음
        roi_g = g[y0:y0+ph, x0:x0+pw]
        gx = cv2.Sobel(roi_g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi_g, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        val = grad[am > 0].mean() if (am > 0).any() else 0.0

        if val < GRAD_MEAN_THR and rng.random() > P_ACCEPT_LOW_GRAD:
            continue

        return x0, y0
    return None


def _mask_to_yolo_seg_lines(mask01: np.ndarray, cls_id: int = 0,
                            min_area: float = 30.0, approx_eps: float = 1.5):
    """
    여러 컨투어(다중 결함) 모두를 YOLOv8-seg 라인으로 출력.
    - min_area: 너무 작은 잡음 제거
    - approx_eps: polygon 단순화(픽셀 단위). 값이 크면 꼭짓점 수가 줄어듦.
    """
    h, w = mask01.shape[:2]
    cnts, _ = cv2.findContours((mask01 > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    lines = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue

        # polygon 단순화 (너무 많은 점 방지)
        peri = cv2.arcLength(c, True)
        eps = max(approx_eps, 0.003 * peri)
        c2 = cv2.approxPolyDP(c, eps, True)

        if c2.shape[0] < 3:
            continue

        x, y, bw, bh = cv2.boundingRect(c2)
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        ww = bw / w
        hh = bh / h

        poly = c2.reshape(-1, 2).astype(np.float32)
        poly[:, 0] = np.clip(poly[:, 0] / w, 0, 1)
        poly[:, 1] = np.clip(poly[:, 1] / h, 0, 1)

        coords = []
        for px, py in poly:
            coords.append(f"{px:.6f}")
            coords.append(f"{py:.6f}")

        line = f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f} " + " ".join(coords)
        lines.append(line)

    return lines


def _post_effects(rng, img_bgr):
    out = img_bgr
    if GLOBAL_STYLE_AUG:
        g = float(rng.uniform(*GAMMA_RANGE))
        lut = np.array([np.clip(((i / 255.0) ** g) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)

        a = float(rng.uniform(*CONTRAST_RANGE))
        b = float(rng.uniform(*BRIGHT_BETA))
        out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        if rng.random() < AUG_BLUR_PROB:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        ns = float(rng.uniform(*AUG_NOISE_STD))
        if ns > 0:
            noise = rng.normal(0, ns, size=out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        q = int(rng.integers(AUG_JPEG_QUALITY[0], AUG_JPEG_QUALITY[1] + 1))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

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


def _p_stain_for_split(split: str):
    if split == "train":
        return P_STAIN_TRAIN
    if split == "valid":
        return P_STAIN_VALID
    return P_STAIN_TEST


def _add_stain_normal(rng, img_bgr: np.ndarray):
    out = img_bgr.copy()
    H, W = out.shape[:2]
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)

    n = int(rng.integers(1, STAIN_MAX_BLOBS + 1))
    for _ in range(n):
        cx = int(rng.integers(0, W))
        cy = int(rng.integers(0, H))
        rx = int(rng.integers(int(W * 0.03), int(W * 0.12)))
        ry = int(rng.integers(int(H * 0.03), int(H * 0.12)))
        ang = float(rng.uniform(0, 180))

        m = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(m, (cx, cy), (rx, ry), ang, 0, 360, 255, -1)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=float(rng.uniform(6, 16)))

        a = (m.astype(np.float32) / 255.0) * float(rng.uniform(STAIN_ALPHA_MIN, STAIN_ALPHA_MAX))
        lab[..., 0] += a * float(rng.uniform(-STAIN_L_SHIFT, STAIN_L_SHIFT))
        lab[..., 1] += a * float(rng.uniform(-STAIN_AB_SHIFT, STAIN_AB_SHIFT))
        lab[..., 2] += a * float(rng.uniform(-STAIN_AB_SHIFT, STAIN_AB_SHIFT))

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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

            if rng.random() < _p_stain_for_split(split):
                stain = _add_stain_normal(rng, tile)
                out_name2 = f"{img_path.stem}__stain.png"
                cv2.imwrite(str(out_img_dir / out_name2), stain)
                (out_lab_dir / f"{Path(out_name2).stem}.txt").write_text("", encoding="utf-8")

        # 2) synthetic defect
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

            # photometric match (weak)
            if PHOTO_MATCH:
                prgb_m = _photometric_match_lab(prgb, roi, alpha)
                prgb = cv2.addWeighted(prgb, 1.0 - PHOTO_STRENGTH, prgb_m, PHOTO_STRENGTH, 0)

            # [OPT] 결함 강도: 과도 대비 억제
            strength = float(rng.uniform(DEFECT_STRENGTH_MIN, DEFECT_STRENGTH_MAX))
            prgb = np.clip(
                roi.astype(np.float32) + strength * (prgb.astype(np.float32) - roi.astype(np.float32)),
                0, 255
            ).astype(np.uint8)

            prgb = _match_sharpness(prgb, roi, alpha)

            pmask2 = pmask.copy()
            if MASK_ERODE > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                pmask2 = cv2.erode(pmask2, k, iterations=MASK_ERODE)

            m01 = (pmask2 > 0).astype(np.uint8)

            de = _deltaE_mean(prgb, roi, m01)
            use_seam = USE_SEAMLESS_CLONE and (de < DE_SEAMLESS_MAX) and (rng.random() < USE_SEAMLESS_PROB)

            if use_seam:
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

        # [OPT] 다중 컨투어 라벨 저장
        lines = _mask_to_yolo_seg_lines(full_mask, OUT_DEFECT_CLS)
        (out_lab_dir / f"{Path(out_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

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
