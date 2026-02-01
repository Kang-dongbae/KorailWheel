# lofo_hero_fixedtest.py
# LOFO (Leave-One-Defect-Group-Out) for HERO wheel few-shot adaptation
# - groups: label-based clustering (seg/bbox supported)
# - For each fold: 1 group is fixed test, rest are train pool
# - K-shot: nested (first k groups from fixed shuffled train pool)
# - BG split: spatial split (top half train, bottom half test) + fixed sampling
# - Repeats: training randomness only (train/test composition fixed per fold)

import shutil
from pathlib import Path
import random
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
import gc
import re
import math
import matplotlib.pyplot as plt

# =========================================================
# [설정]
# =========================================================
ROOT_DIR = Path(r"C:\Dev\KorailWheel")
SOURCE_DIR = ROOT_DIR / "data" / "data_tiles" / "external_test"
WORK_DIR = ROOT_DIR / "data" / "paper_experiment_lofo"
RUNS_DIR = ROOT_DIR / "runs" / "paper_final_lofo"

PRETRAINED_WEIGHTS = ROOT_DIR / "runs" / "synth_yolo" / "internal" / "train" / "weights" / "best.pt"

HERO_WHEEL = "L0316"

LABEL_FORMAT = "seg"   # "seg" or "bbox"
DEFECT_CLASS_ID = 0

# K-shot (group count)
ADAPT_K_SHOTS = [0, 1, 2, 3, 4, 5]
N_REPEATS = 10

# Clustering eps (global px)
DBSCAN_EPS_PX = 25

# Tile coord grid size (must match tiling)
TILE_SIZE = 512

# BG sampling (fixed)
TRAIN_BG_N = 30
TEST_BG_N = 30

# Training
EPOCHS = 30
IMGSZ = 512
BATCH = 4
WORKERS = 0

# Seeds
FOLD_SHUFFLE_SEED = 123   # controls fold-internal train pool order
BG_TEST_SEED = 777
BG_TRAIN_SEED = 778

# =========================================================
# [유틸]
# =========================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_wheel_id(filename: str) -> str:
    return filename.split('_')[0]

def parse_tile_coords(filename: str):
    mx = re.search(r"_x(\d+)", filename)
    my = re.search(r"_y(\d+)", filename)
    x = int(mx.group(1)) if mx else 0
    y = int(my.group(1)) if my else 0
    return x, y

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# =========================================================
# [라벨 파싱] seg: polygon centroid(면적가중) + 가장 큰 폴리곤 1개
# =========================================================
def polygon_area_and_centroid(xs, ys):
    n = len(xs)
    if n < 3:
        cx = float(sum(xs)) / float(len(xs))
        cy = float(sum(ys)) / float(len(ys))
        return 0.0, cx, cy

    x = xs + [xs[0]]
    y = ys + [ys[0]]

    A = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(n):
        cross = x[i] * y[i+1] - x[i+1] * y[i]
        A += cross
        Cx += (x[i] + x[i+1]) * cross
        Cy += (y[i] + y[i+1]) * cross

    A *= 0.5
    area_abs = abs(A)

    if abs(A) < 1e-9:
        cx = float(sum(xs)) / float(len(xs))
        cy = float(sum(ys)) / float(len(ys))
        return 0.0, cx, cy

    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return area_abs, Cx, Cy

def parse_tile_representative_center(lbl_path: Path, label_format: str, defect_class_id: int):
    txt = lbl_path.read_text(encoding="utf-8").strip()
    if not txt:
        return None

    if label_format == "bbox":
        for line in txt.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            if cls != defect_class_id:
                continue
            cx = float(parts[1]); cy = float(parts[2])
            return (cx, cy)
        return None

    best = None  # (area, cx, cy)
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        if cls != defect_class_id:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) < 6 or (len(coords) % 2 != 0):
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        area, cx, cy = polygon_area_and_centroid(xs, ys)
        if best is None or area > best[0]:
            best = (area, cx, cy)

    if best is None:
        return None
    return (best[1], best[2])

def get_global_center_for_tile(img_path: Path, lbl_dir: Path,
                               label_format: str, defect_class_id: int,
                               tile_size: int):
    lbl_path = lbl_dir / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        return None

    center = parse_tile_representative_center(lbl_path, label_format, defect_class_id)
    if center is None:
        return None

    tile_x, tile_y = parse_tile_coords(img_path.name)
    cx, cy = center
    gx = tile_x + cx * tile_size
    gy = tile_y + cy * tile_size
    return (gx, gy)

# =========================================================
# [클러스터링] eps-graph connected components
# =========================================================
def cluster_points_bfs(points, eps):
    n = len(points)
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        member_imgs = [points[i][0]]

        while queue:
            cur = queue.pop(0)
            cur_pt = points[cur][1]
            for j in range(n):
                if visited[j]:
                    continue
                if calculate_distance(cur_pt, points[j][1]) < eps:
                    visited[j] = True
                    queue.append(j)
                    member_imgs.append(points[j][0])

        uniq = list(dict.fromkeys(member_imgs))
        groups.append(uniq)

    return groups

def group_tiles_by_label_clustering(img_files, lbl_dir: Path,
                                    label_format: str, defect_class_id: int,
                                    eps: float, tile_size: int):
    wheel_dict = {}
    for f in img_files:
        wheel_dict.setdefault(get_wheel_id(f.name), []).append(f)

    all_groups = []
    for _, files in wheel_dict.items():
        points = []
        for img_path in files:
            c = get_global_center_for_tile(
                img_path, lbl_dir,
                label_format=label_format,
                defect_class_id=defect_class_id,
                tile_size=tile_size
            )
            if c is not None:
                points.append((img_path, c))

        if points:
            all_groups.extend(cluster_points_bfs(points, eps))

    return [g for g in all_groups if g]

# =========================================================
# [파일 관리]
# =========================================================
def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    for split in ["train", "valid", "test"]:
        (path / split / "images").mkdir(parents=True, exist_ok=True)
        (path / split / "labels").mkdir(parents=True, exist_ok=True)

def copy_files(file_list, dst_root: Path, split: str, lbl_source_dir: Path):
    dst_img = dst_root / split / "images"
    dst_lbl = dst_root / split / "labels"
    for src in file_list:
        shutil.copy(src, dst_img / src.name)
        src_lbl = lbl_source_dir / f"{src.stem}.txt"
        dst_lbl_path = dst_lbl / f"{src.stem}.txt"
        if src_lbl.exists():
            shutil.copy(src_lbl, dst_lbl_path)
        else:
            dst_lbl_path.write_text("", encoding="utf-8")

def write_yaml(work_dir: Path, train_path: str, val_path: str, test_path: str):
    abs_path = work_dir.resolve().as_posix()
    (work_dir / "data.yaml").write_text(
        f"""path: {abs_path}
train: {train_path}
val: {val_path}
test: {test_path}
nc: 1
names: ["defect"]
""",
        encoding="utf-8",
    )

# =========================================================
# [LOFO 실행]
# =========================================================
def run_lofo():
    defect_img_dir = SOURCE_DIR / "defect" / "images"
    defect_lbl_dir = SOURCE_DIR / "defect" / "labels"
    normal_img_dir = SOURCE_DIR / "normal" / "images"
    normal_lbl_dir = SOURCE_DIR / "normal" / "labels"

    hero_defects = [p for p in defect_img_dir.glob(f"{HERO_WHEEL}*.*") if p.suffix.lower() in [".jpg", ".png"]]
    hero_normals = [p for p in normal_img_dir.glob(f"{HERO_WHEEL}*.*") if p.suffix.lower() in [".jpg", ".png"]]

    groups = group_tiles_by_label_clustering(
        hero_defects, defect_lbl_dir,
        label_format=LABEL_FORMAT,
        defect_class_id=DEFECT_CLASS_ID,
        eps=DBSCAN_EPS_PX,
        tile_size=TILE_SIZE
    )

    # 그룹 인덱스 고정(재현성)
    # fold마다 test는 group i
    n_groups = len(groups)
    if n_groups < 2:
        return

    # BG: spatial split
    hero_normals.sort(key=lambda p: parse_tile_coords(p.name)[1])
    mid = len(hero_normals) // 2
    bg_pool_train = hero_normals[:mid]
    bg_pool_test = hero_normals[mid:]

    seed_all(BG_TEST_SEED)
    test_bg = random.sample(bg_pool_test, min(TEST_BG_N, len(bg_pool_test)))

    seed_all(BG_TRAIN_SEED)
    train_bg_fixed = random.sample(bg_pool_train, min(TRAIN_BG_N, len(bg_pool_train)))

    (RUNS_DIR / "adapt").mkdir(parents=True, exist_ok=True)
    reset_dir(WORK_DIR)

    results = []

    for fold in range(n_groups):
        # fold 구성: test = groups[fold], train_pool = others
        test_groups = [groups[fold]]
        train_pool_groups = [g for i, g in enumerate(groups) if i != fold]

        # train_pool 순서 고정
        rng = random.Random(FOLD_SHUFFLE_SEED + fold)
        train_pool_groups = train_pool_groups.copy()
        rng.shuffle(train_pool_groups)

        test_files = [f for g in test_groups for f in g]

        for k in ADAPT_K_SHOTS:
            if k > len(train_pool_groups):
                continue

            # k=0이면 defect train 그룹 없음
            if k == 0:
                train_files_k = []
            else:
                train_groups_k = train_pool_groups[:k]
                train_files_k = [f for g in train_groups_k for f in g]

            for r in range(N_REPEATS):
                gc.collect(); torch.cuda.empty_cache()

                train_seed = 9000 + 1000 * fold + 100 * k + r
                seed_all(train_seed)

                reset_dir(WORK_DIR)

                # always copy BG into train/valid
                copy_files(train_bg_fixed, WORK_DIR, "train", normal_lbl_dir)
                copy_files(train_bg_fixed, WORK_DIR, "valid", normal_lbl_dir)

                # k>0일 때만 defect를 train/valid에 추가
                if k > 0:
                    copy_files(train_files_k, WORK_DIR, "train", defect_lbl_dir)
                    copy_files(train_files_k, WORK_DIR, "valid", defect_lbl_dir)

                # test는 항상 동일
                copy_files(test_files, WORK_DIR, "test", defect_lbl_dir)
                copy_files(test_bg, WORK_DIR, "test", normal_lbl_dir)

                write_yaml(WORK_DIR, "train/images", "valid/images", "test/images")

                model = YOLO(str(PRETRAINED_WEIGHTS))

                # k>0일 때만 fine-tune
                if k > 0:
                    model.train(
                        data=str(WORK_DIR / "data.yaml"),
                        epochs=EPOCHS,
                        imgsz=IMGSZ,
                        batch=BATCH,
                        project=str(RUNS_DIR / "adapt"),
                        name=f"lofo_{HERO_WHEEL}_fold{fold}_k{k}_r{r}",
                        exist_ok=True,
                        verbose=False,
                        workers=WORKERS,
                        seed=train_seed,
                        deterministic=True,
                        amp=False,
                    )
                else:
                    # k=0은 학습 없음. 저장 폴더 이름만 구분하고 싶으면 아래 로그만 남겨도 됨.
                    pass

                m = model.val(data=str(WORK_DIR / "data.yaml"), split="test", verbose=False)
                recall = float(m.box.r.mean()) if hasattr(m.box.r, "mean") else float(m.box.r)
                map50 = float(m.box.map50)

                results.append({
                    "wheel": HERO_WHEEL,
                    "label_format": LABEL_FORMAT,
                    "defect_class_id": DEFECT_CLASS_ID,
                    "eps": DBSCAN_EPS_PX,
                    "tile_size": TILE_SIZE,
                    "imgsz": IMGSZ,
                    "fold": fold,
                    "n_groups_total": n_groups,
                    "k_shot": k,
                    "round": r,
                    "train_seed": train_seed,
                    "recall": recall,
                    "map50": map50,
                    "n_train_pool_groups": len(train_pool_groups),
                    "n_train_def_tiles": len(train_files_k),
                    "n_test_def_tiles": len(test_files),
                    "n_train_bg_tiles": len(train_bg_fixed),
                    "n_test_bg_tiles": len(test_bg),
                })
                del model


    df = pd.DataFrame(results)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RUNS_DIR / "hero_lofo.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if df.empty:
        return

    # fold 평균 -> k별 전체 평균
    # (fold별 mean을 먼저 낸 다음, fold 간 평균/표준편차를 내면 더 깔끔)
    fold_mean = df.groupby(["fold", "k_shot"])[["recall", "map50"]].mean().reset_index()
    summary = fold_mean.groupby("k_shot")[["recall", "map50"]].agg(["mean", "std"]).reset_index()
    summary.to_csv(RUNS_DIR / "hero_lofo_summary.csv", index=False, encoding="utf-8-sig")

    xs = summary["k_shot"].values

    plt.figure()
    plt.errorbar(xs, summary[("map50","mean")].values, yerr=summary[("map50","std")].values, marker="o", capsize=5)
    plt.xlabel("K-shot")
    plt.ylabel("mAP@0.5")
    plt.title("LOFO Few-shot Adaptation (Fixed BG, Group-wise Holdout)")
    plt.grid(True, alpha=0.3)
    plt.savefig(RUNS_DIR / "lofo_curve_map50.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.errorbar(xs, summary[("recall","mean")].values, yerr=summary[("recall","std")].values, marker="o", capsize=5)
    plt.xlabel("K-shot")
    plt.ylabel("Recall")
    plt.title("LOFO Few-shot Adaptation (Fixed BG, Group-wise Holdout)")
    plt.grid(True, alpha=0.3)
    plt.savefig(RUNS_DIR / "lofo_curve_recall.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_lofo()
