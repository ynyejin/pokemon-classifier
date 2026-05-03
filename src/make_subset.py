import os
import shutil
from pathlib import Path

SOURCE_DIR = Path("data/merged_raw")
OUTPUT_DIR = Path("data/merged_300")

NUM_CLASSES = 300
MAX_IMAGES_PER_CLASS = 100  # 클래스당 최대 이미지 수


def make_subset():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 이미지가 많은 클래스 순으로 정렬
    class_infos = []

    for class_dir in SOURCE_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]

        if len(images) >= 10:  # 너무 적은 클래스 제외
            class_infos.append((class_dir.name, images))

    class_infos.sort(key=lambda x: len(x[1]), reverse=True)

    selected = class_infos[:NUM_CLASSES]

    for class_name, images in selected:
        dst_class_dir = OUTPUT_DIR / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(images[:MAX_IMAGES_PER_CLASS]):
            ext = img_path.suffix.lower()
            dst_path = dst_class_dir / f"{class_name}_{i}{ext}"
            shutil.copy2(img_path, dst_path)

    print(f"완료: {len(selected)}개 클래스 생성")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    make_subset()