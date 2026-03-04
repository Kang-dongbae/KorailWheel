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
from collections import deque, defaultdict
from functools import lru_cache

# =========================
# 설정
# =========================
ROOT_DIR = Path(r"C:\Dev\KorailWheel")
SOURCE_DIR = ROOT_DIR / "data" / "data_tiles" / "external_test"
WORK_DIR = ROOT_DIR / "data" / "paper_experiment_v7_B1"
RUNS_DIR = ROOT_DIR / "runs" / "b1_baseline1"

PRETRAINED_WEIGHTS = ROOT_DIR / "runs" / "synth_yolo" / "internal" / "train" / "weights" / "best.pt"

HERO_WHEEL = "L0316"
LABEL_FORMAT = "seg"
DEFECT_CLASS_ID = 0

HERO_HOLDOUT_DEFECTS = 1
HERO_SPLIT_SEED = 123

DBSCAN_EPS_PX = 25
TILE_SIZE = 512

TRAIN_BG_N = 30
TEST_BG_N = 30

EPOCHS = 30
IMGSZ = 512
BATCH = 4
WORKERS = 0
AMP = False

K_SHOTS = [1, 2, 3]
N_REPEATS = 10

X_RE = re.compile(r"_x(\d+)")
Y_RE = re.compile(r"_y(\d+)")

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_wheel_id(filename: str) -> str:
    return filename.split('_')[0]

@lru_cache(maxsize=20000)
def parse_tile_coords(filename: str):
    mx = X_RE.search(filename)
    my = Y_RE.search(filename)
    x = int(mx.group(1)) if mx else 0
    y = int(my.group(1)) if my else 0
    return x, y

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
    best = None
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
    return None if best is None else (best[1], best[2])

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
    return (tile_x + cx * tile_size, tile_y + cy * tile_size)

def cluster_points_bfs(points, eps):
    # grid + deque + squared distance
    if not points:
        return []
    eps2 = eps * eps
    grid = defaultdict(list)
    for idx, (_, (x, y)) in enumerate(points):
        grid[(int(x // eps), int(y // eps))].append(idx)

    visited = [False] * len(points)
    groups = []

    for i in range(len(points)):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        member_imgs = []
        while q:
            cur = q.popleft()
            img_path, (cx, cy) = points[cur]
            member_imgs.append(img_path)

            gx, gy = int(cx // eps), int(cy // eps)
            for nx in (gx - 1, gx, gx + 1):
                for ny in (gy - 1, gy, gy + 1):
                    for j in grid.get((nx, ny), []):
                        if visited[j]:
                            continue
                        _, (px, py) = points[j]
                        dx = cx - px
                        dy = cy - py
                        if (dx * dx + dy * dy) < eps2:
                            visited[j] = True
                            q.append(j)

        groups.append(list(dict.fromkeys(member_imgs)))
    return groups

def group_tiles_by_label_clustering(img_files, lbl_dir: Path,
                                    label_format: str, defect_class_id: int,
                                    eps: float, tile_size: int):
    wheel_dict = {}
    for f in img_files:
        wheel_dict.setdefault(get_wheel_id(f.name), []).append(f)
    all_groups = []
    for files in wheel_dict.values():
        points = []
        for img_path in files:
            c = get_global_center_for_tile(img_path, lbl_dir, label_format, defect_class_id, tile_size)
            if c is not None:
                points.append((img_path, c))
        if points:
            all_groups.extend(cluster_points_bfs(points, eps))
    return [g for g in all_groups if g]

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
        shutil.copy2(src, dst_img / src.name)
        src_lbl = lbl_source_dir / f"{src.stem}.txt"
        dst_lbl_path = dst_lbl / f"{src.stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl_path)
        else:
            dst_lbl_path.write_text("", encoding="utf-8")

def remove_files(file_list, dst_root: Path, split: str):
    dst_img = dst_root / split / "images"
    dst_lbl = dst_root / split / "labels"
    for src in file_list:
        p_img = dst_img / src.name
        p_lbl = dst_lbl / f"{src.stem}.txt"
        if p_img.exists():
            p_img.unlink()
        if p_lbl.exists():
            p_lbl.unlink()

def write_yaml(work_dir: Path):
    abs_path = work_dir.resolve().as_posix()
    (work_dir / "data.yaml").write_text(
        f"""path: {abs_path}
train: train/images
val: valid/images
test: test/images
nc: 1
names: ["defect"]
""",
        encoding="utf-8",
    )

def run_B1_random_kshot():
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
    if len(groups) < (HERO_HOLDOUT_DEFECTS + 1):
        print("Not enough defect groups. abort.")
        return

    hero_normals.sort(key=lambda p: parse_tile_coords(p.name)[1])
    mid = len(hero_normals) // 2
    bg_pool_train = hero_normals[:mid]
    bg_pool_test = hero_normals[mid:]

    seed_all(HERO_SPLIT_SEED)
    g2 = groups.copy()
    random.shuffle(g2)
    test_groups = g2[:HERO_HOLDOUT_DEFECTS]
    train_pool_groups = g2[HERO_HOLDOUT_DEFECTS:]
    test_def_files = [f for g in test_groups for f in g]

    seed_all(777)
    test_bg = random.sample(bg_pool_test, min(TEST_BG_N, len(bg_pool_test)))
    seed_all(778)
    train_bg_fixed = random.sample(bg_pool_train, min(TRAIN_BG_N, len(bg_pool_train)))

    (RUNS_DIR / "B1_random").mkdir(parents=True, exist_ok=True)

    # 고정 데이터는 1회만 복사
    reset_dir(WORK_DIR)
    copy_files(train_bg_fixed, WORK_DIR, "train", normal_lbl_dir)
    copy_files(train_bg_fixed, WORK_DIR, "valid", normal_lbl_dir)
    copy_files(test_def_files, WORK_DIR, "test", defect_lbl_dir)
    copy_files(test_bg, WORK_DIR, "test", normal_lbl_dir)
    write_yaml(WORK_DIR)

    results = []
    prev_train_def_files = []

    for k in K_SHOTS:
        if k > len(train_pool_groups):
            continue

        for r in range(N_REPEATS):
            if (r % 3) == 0:
                gc.collect()
                torch.cuda.empty_cache()

            pick_seed = 50000 + 1000 * k + r
            rng = random.Random(pick_seed)
            train_groups_k = rng.sample(train_pool_groups, k)
            train_def_files = [f for g in train_groups_k for f in g]

            # 바뀐 결함 파일만 증분 반영
            prev_set = set(prev_train_def_files)
            cur_set = set(train_def_files)
            to_remove = list(prev_set - cur_set)
            to_add = list(cur_set - prev_set)

            if to_remove:
                remove_files(to_remove, WORK_DIR, "train")
                remove_files(to_remove, WORK_DIR, "valid")
            if to_add:
                copy_files(to_add, WORK_DIR, "train", defect_lbl_dir)
                copy_files(to_add, WORK_DIR, "valid", defect_lbl_dir)

            prev_train_def_files = train_def_files

            train_seed = 9000 + 100 * k + r
            seed_all(train_seed)

            model = YOLO(str(PRETRAINED_WEIGHTS))
            model.train(
                data=str(WORK_DIR / "data.yaml"),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH,
                project=str(RUNS_DIR / "B1_random"),
                name=f"B1_{HERO_WHEEL}_k{k}_r{r}",
                exist_ok=True,
                verbose=False,
                workers=WORKERS,
                seed=train_seed,
                deterministic=True,
                amp=AMP,
            )

            m = model.val(
                data=str(WORK_DIR / "data.yaml"),
                split="test",
                imgsz=IMGSZ,
                batch=BATCH,
                verbose=False,
                workers=WORKERS,
            )
            recall = float(m.box.r.mean()) if hasattr(m.box.r, "mean") else float(m.box.r)
            map50 = float(m.box.map50)

            results.append({
                "wheel": HERO_WHEEL,
                "mode": "B1_random_kshot",
                "eps": DBSCAN_EPS_PX,
                "k_shot": k,
                "round": r,
                "pick_seed": pick_seed,
                "train_seed": train_seed,
                "imgsz": IMGSZ,
                "recall": recall,
                "map50": map50,
                "n_def_groups_total": len(groups),
                "n_train_pool_groups": len(train_pool_groups),
                "n_train_def_tiles": len(train_def_files),
                "n_test_def_tiles": len(test_def_files),
            })
            del model

    df = pd.DataFrame(results)
    out_csv = RUNS_DIR / "B1_random_kshot.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if df.empty:
        print("Empty results.")
        return

    summary = df.groupby("k_shot")[["recall", "map50"]].agg(["mean", "std"])
    out_sum = RUNS_DIR / "B1_random_kshot_summary.csv"
    summary.to_csv(out_sum, encoding="utf-8-sig")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_sum}")
    print(summary)

if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_B1_random_kshot()
