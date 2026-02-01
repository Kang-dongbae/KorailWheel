import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# =========================================================
# [설정] 경로를 본인 환경에 맞게 확인하세요!
# =========================================================
ROOT_DIR = Path(r"C:\Dev\KorailWheel\data")

# 1. 원본: 실제 결함 8장이 있는 폴더 (이미지 + YOLO 라벨 txt)
REAL_IMG_DIR = ROOT_DIR / "data_tiles" / "external_test" / "defect" / "images"
REAL_LAB_DIR = ROOT_DIR / "data_tiles" / "external_test" / "defect" / "labels"

# 2. 출력: 리얼 패치가 저장될 곳
OUT_DIR = ROOT_DIR / "patch_real" / "train"
OUT_IMG_DIR = OUT_DIR / "images"
OUT_LAB_DIR = OUT_DIR / "labels"

# 3. 옵션
MARGIN = 2    # 결함 주변 여유 픽셀
MIN_AREA = 20 # 너무 작은 조각 무시
# =========================================================

def crop_real_patches_v2():
    # 1. 폴더 초기화
    if OUT_DIR.exists():
        try:
            shutil.rmtree(OUT_DIR)
        except Exception as e:
            print(f"폴더 삭제 중 오류 (무시 가능): {e}")
            
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LAB_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = list(REAL_IMG_DIR.glob("*.png")) + list(REAL_IMG_DIR.glob("*.jpg"))
    print(f"▶ 원본 이미지 {len(img_paths)}장에서 결함 추출을 시작합니다...")

    total_patches = 0

    for p in tqdm(img_paths, desc="Extraction"):
        img = cv2.imread(str(p))
        if img is None: continue
        h_img, w_img = img.shape[:2]

        lab_path = REAL_LAB_DIR / f"{p.stem}.txt"
        if not lab_path.exists():
            continue

        with open(lab_path, 'r') as f:
            lines = f.readlines()

        # 한 이미지 안에 있는 여러 결함들을 각각 처리
        for i, line in enumerate(lines):
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5: continue

            # (1) 폴리곤 좌표 복원
            poly_norm = np.array(parts[5:]).reshape(-1, 2)
            poly_px = (poly_norm * [w_img, h_img]).astype(np.int32)

            # (2) 마스크 생성 (결함 모양대로 따내기 위함)
            # 전체 이미지 크기의 블랙 마스크 생성
            full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(full_mask, [poly_px], 255) # 결함 부위만 흰색

            # (3) Bounding Box 계산
            x, y, w, h = cv2.boundingRect(poly_px)
            
            # 마진 적용
            x1 = max(0, x - MARGIN)
            y1 = max(0, y - MARGIN)
            x2 = min(w_img, x + w + MARGIN)
            y2 = min(h_img, y + h + MARGIN)

            if (x2 - x1) < 5 or (y2 - y1) < 5: continue

            # (4) 이미지 Crop & 배경 제거 (핵심 수정 부분)
            # 원본 이미지에서 사각형으로 잘라냄
            img_crop = img[y1:y2, x1:x2].copy()
            # 마스크도 똑같이 잘라냄
            mask_crop = full_mask[y1:y2, x1:x2]

            # [중요] 마스크가 0(배경)인 부분은 이미지도 0(검정)으로 만듦
            # 이렇게 해야 '정상 휠 배경'이 패치에 포함되지 않음
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)

            # (5) 패치 기준 라벨(Polygon) 재계산 (synthesis.py용)
            ph, pw = img_crop.shape[:2]
            poly_crop = poly_px - [x1, y1] # 좌표 이동
            
            # 정규화
            poly_crop_norm = poly_crop.astype(np.float32) / [pw, ph]
            poly_crop_norm[:, 0] = np.clip(poly_crop_norm[:, 0], 0, 1)
            poly_crop_norm[:, 1] = np.clip(poly_crop_norm[:, 1], 0, 1)

            # (6) 파일 저장
            save_name = f"{p.stem}_defect_{i:02d}" # 이름 겹치지 않게 인덱스 추가
            
            # 배경이 검은색인 패치 이미지 저장
            cv2.imwrite(str(OUT_IMG_DIR / f"{save_name}.png"), img_crop)

            # 라벨 파일 저장
            bx, by, bw, bh = cv2.boundingRect(poly_crop)
            n_xc = (bx + bw / 2) / pw
            n_yc = (by + bh / 2) / ph
            n_w  = bw / pw
            n_h  = bh / ph

            coords_str = " ".join([f"{val:.6f}" for val in poly_crop_norm.flatten()])
            label_line = f"0 {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f} {coords_str}"

            with open(OUT_LAB_DIR / f"{save_name}.txt", "w") as f_out:
                f_out.write(label_line)

            total_patches += 1

    print(f"\n[완료] 총 {total_patches}개의 '배경 제거된 리얼 패치' 생성!")
    print(f"저장 위치: {OUT_DIR}")
    print("★ 확인: 생성된 이미지를 열어보세요. 결함 주변이 까맣게 되어 있어야 정상입니다.")

if __name__ == "__main__":
    crop_real_patches_v2()