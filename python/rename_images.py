import os
from glob import glob

img_dir = "hand_images"

jpg_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
total = len(jpg_files)

# 第一步：全部重命名为临时名，彻底避免冲突
for i, path in enumerate(jpg_files):
    tmp = os.path.join(img_dir, f"__tmp_{i:06d}.jpg")
    os.rename(path, tmp)

# 第二步：从临时名改为目标名
tmp_files = sorted(glob(os.path.join(img_dir, "__tmp_*.jpg")))
for i, path in enumerate(tmp_files, start=1):
    new_path = os.path.join(img_dir, f"hand_{i:04d}.jpg")
    os.rename(path, new_path)
    print(f"→ hand_{i:04d}.jpg")

print(f"\n完成，共重命名 {total} 张")