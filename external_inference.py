# external_inference.py
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# =========================
# FIXED PATHS (USER)
# =========================
EXT_ROOT  = Path(r"C:\Dev\KorailWheel\data\data_tiles\external_test")
RUNS_ROOT = Path(r"C:\Dev\KorailWheel\runs\synth_yolo")

# weights produced by train_yolo.py
#WEIGHTS = RUNS_ROOT / "internal" / "train" / "weights" / "best.pt"
WEIGHTS = RUNS_ROOT / "finetune" / "final_simple" / "weights" / "best.pt"
# =========================
# INFER SETTINGS
# =========================
IMG_SIZE = 640          # train_yolo.py와 동일하게
RAW_CONF = 0.001        # 모델 호출은 낮게 (거의 다 뽑기)
POST_CONF = 0.005        # 후처리 임계값(여기만 바꿔가며 실험)
IOU     = 0.5
MAX_DET = 100           # 10은 너무 작음(외부에서 희귀하게 뜨는 것까지 잘림)
MIN_AREA_PX = 5        # 잡음(1~수픽셀) 제거용
DEVICE   = "0"         # "cpu" or "0"



IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _img_list(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def _pred_best_mask_and_score(model: YOLO, img_path: Path):
    res = model.predict(
        source=str(img_path),
        task="segment",
        imgsz=IMG_SIZE,
        conf=RAW_CONF,
        iou=IOU,
        max_det=MAX_DET,
        device=DEVICE,
        retina_masks=True,
        verbose=False,
        #augment=True,
    )[0]

    h, w = res.orig_shape
    out = np.zeros((h, w), dtype=np.uint8)

    if res.masks is None or res.boxes is None or res.boxes.conf is None or len(res.boxes.conf) == 0:
        return out, 0.0, 0, 0

    confs = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
    num_inst = int(len(confs))
    best_conf = float(confs.max())

    md = res.masks.data
    if md is None:
        return out, best_conf, num_inst, 0

    md = md.detach().cpu().numpy()  # (N,H,W)
    if not (md.ndim == 3 and md.shape[1] == h and md.shape[2] == w):
        return out, best_conf, num_inst, 0

    # conf/area 조건 통과한 것만 union
    best_area = 0
    for i, c in enumerate(confs):
        if float(c) < POST_CONF:
            continue
        m = (md[i] > 0).astype(np.uint8)
        area = int(m.sum())
        if area < MIN_AREA_PX:
            continue
        out = np.maximum(out, m)
        best_area = max(best_area, area)

    return out, best_conf, num_inst, best_area





def _overlay_pred(img_bgr, pred01):
    vis = img_bgr.copy()
    pr_c, _ = cv2.findContours((pred01 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, pr_c, -1, (0, 0, 255), 2)
    return vis


def infer_split(model: YOLO, cls_name: str):
    in_dir = EXT_ROOT / cls_name / "images"
    if not in_dir.exists():
        print("[SKIP] not found:", in_dir)
        return

    out_dir = RUNS_ROOT / "external" / f"infer_{cls_name}"
    ov_dir  = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in tqdm(_img_list(in_dir), desc=f"external:{cls_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        pred, score, num_inst, best_area = _pred_best_mask_and_score(model, img_path)
        pred_area_frac = float(pred.sum() / (h * w))

        rows.append({
            "image": str(img_path),
            "score": score,                 # best conf (POST_CONF 전/후 판단에 사용)
            "num_inst": int(num_inst),      # RAW_CONF에서 몇 개가 떴는지
            "best_area": int(best_area),    # best 인스턴스 면적
            "pred_area_frac": pred_area_frac,
            "pred_pixels": int(pred.sum()),
        })


        vis = _overlay_pred(img, pred)
        cv2.imwrite(str(ov_dir / f"{img_path.stem}.jpg"), vis)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{cls_name}_pred.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] {cls_name}: {len(df)} -> {out_dir}")


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"best.pt not found: {WEIGHTS}\n먼저 python train_yolo.py 실행하세요.")

    model = YOLO(str(WEIGHTS))
    infer_split(model, "normal")
    infer_split(model, "defect")


if __name__ == "__main__":
    main()
