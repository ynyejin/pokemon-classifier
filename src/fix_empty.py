import os

def remove_empty_dirs(root_dir):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(root_dir, split)

        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)

            if os.path.isdir(cls_path):
                if len(os.listdir(cls_path)) == 0:
                    print(f"삭제: {cls_path}")
                    os.rmdir(cls_path)

remove_empty_dirs("data/processed")
print("✅ 빈 폴더 정리 완료!")