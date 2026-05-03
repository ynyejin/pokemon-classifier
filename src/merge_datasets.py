import os
import shutil
from pathlib import Path

DATASET_DIRS = [
    "data/PokemonData_32k",
    "data/PokemonData_7k",
]

OUTPUT_DIR = "data/merged_raw"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_class_name(name):
    # 클래스명 통일: pikachu, Pikachu, PIKACHU -> Pikachu
    return name.strip().replace("_", " ").replace("-", " ").title().replace(" ", "")


def merge_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset_idx, dataset_dir in enumerate(DATASET_DIRS):
        dataset_path = Path(dataset_dir)

        for class_folder in dataset_path.iterdir():
            if not class_folder.is_dir():
                continue

            class_name = normalize_class_name(class_folder.name)
            output_class_dir = Path(OUTPUT_DIR) / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            count = len(list(output_class_dir.glob("*")))

            # 하위 폴더까지 이미지 전부 찾기
            for img_path in class_folder.rglob("*"):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                new_name = f"ds{dataset_idx}_{class_name}_{count}{img_path.suffix.lower()}"
                dst_path = output_class_dir / new_name

                shutil.copy2(img_path, dst_path)
                count += 1

    print("✅ Dataset merge 완료!")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    merge_datasets()