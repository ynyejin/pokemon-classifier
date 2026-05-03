import os
import random
import shutil

def split_dataset(
    raw_dir="data/raw",
    output_dir="data/processed",
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    random.seed(seed)

    classes = os.listdir(raw_dir)

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in splits.items():
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)

            for f in files:
                src = os.path.join(cls_path, f)
                dst = os.path.join(split_dir, f)
                shutil.copy(src, dst)

    print("Dataset split 완료!")