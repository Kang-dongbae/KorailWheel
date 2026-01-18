# config.py
from pathlib import Path

ROOT = Path(r"C:\Dev\KorailWheel")

DATA = ROOT / "data" / "st1_roi"
DATA_YAML = DATA / "data.yaml" 
TEST_IMAGES = DATA / "test" / "images"
TEST_LABELS = DATA / "test" / "labels"

RUNS_DIR = ROOT / "runs"
ART_DIR = RUNS_DIR / "artifacts" / "stage1"
