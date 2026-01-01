import os
import torch

# ==========================================
# 1. PATHS
# ==========================================
PROJECT_ROOT = r"C:\Dev\KorailWheel"

# 원본 데이터 경로 (crop_wheel)
RAW_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "crop_wheel")

# 전처리 결과물 경로
# 1차: Band ROI (QC 결과 포함)
OUT_BAND_ROOT = os.path.join(PROJECT_ROOT, "data", "band_wheel")
# 2차: Tiling (최종 학습용)
OUT_TILE_ROOT = os.path.join(PROJECT_ROOT, "data", "band_tiles")

# 결과 저장 (Checkpoint, Logs)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# ==========================================
# 2. PREPROCESSING PARAMS (notebook3)
# ==========================================
RESIZE_HW = (512, 256)  # (H, W) processing size
BAND_W = 72             # Band width
TILE_H = 256            # Tile height
STRIDE = 128            # Tile stride
QC_PASS_DIR_NAME = "images" 

# ==========================================
# 3. AE TRAIN PARAMS (main.py)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

AE_IMG_SIZE = 256
AE_BATCH_SIZE = 32
AE_EPOCHS = 20
AE_LR = 1e-3

# Loss & Masking
MASK_THRESHOLD = 0.02
BG_WEIGHT = 0.02

# Evaluation
REQUIRE_TILES = 3      # None or int
AGG_METHOD = "max"     # 'max' or 'mean'
VAL_Q = 0.995          # Threshold quantile
TOPK_VIS = 10          # Number of visualization images

# ==========================================
# 4. UTILS
# ==========================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)