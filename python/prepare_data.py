import json
import numpy as np
from glob import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORDER = [
    'wrist',
    'thumb1',  'thumb2',  'thumb3',  'thumb4',
    'index1',  'index2',  'index3',  'index4',
    'middle1', 'middle2', 'middle3', 'middle4',
    'ring1',   'ring2',   'ring3',   'ring4',
    'pinky1',  'pinky2',  'pinky3',  'pinky4',
]

SKIP_THRESHOLD = 2


def load_annotations(json_dir, img_dir):
    data = []
    json_files = sorted(glob(os.path.join(json_dir, "*.json")))

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)

        if len(anno['shapes']) == 0:
            print(f"跳过（无标注）: {os.path.basename(json_path)}")
            continue

        img_w = anno['imageWidth']
        img_h = anno['imageHeight']
        coord_type = anno.get("coord_type", "pixel")   # 默认当像素坐标处理

        points = {}
        for shape in anno['shapes']:
            label = shape['label']
            x, y = shape['points'][0]

            if coord_type == "normalized_0_1":
                nx, ny = float(x), float(y)
            else:
                # pixel → 归一化
                nx = float(x) / img_w
                ny = float(y) / img_h

            nx = float(np.clip(nx, 0.0, 1.0))
            ny = float(np.clip(ny, 0.0, 1.0))
            points[label] = [nx, ny]

        missing = [name for name in ORDER if name not in points]
        if len(missing) > SKIP_THRESHOLD:
            print(f"跳过（缺失 {len(missing)} 个关键点）: {os.path.basename(json_path)}")
            continue
        if missing:
            print(f"警告（缺失 {len(missing)} 个）: {os.path.basename(json_path)}")

        coords = []
        for name in ORDER:
            if name in points:
                coords.extend(points[name])
            else:
                coords.extend([0.5, 0.5])

        # 找图片路径
        json_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            test_path = os.path.join(img_dir, json_name + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break

        if img_path is None:
            img_name = os.path.basename(anno['imagePath'])
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"警告: 找不到图片，跳过: {img_path}")
                continue

        data.append({
            'image':  img_path,
            'coords': np.array(coords, dtype=np.float32),
        })

    return data


if __name__ == '__main__':
    labels_dir = os.path.join(BASE_DIR, "labels")
    images_dir = os.path.join(BASE_DIR, "hand_images")

    train_data = load_annotations(labels_dir, images_dir)

    if len(train_data) == 0:
        print("错误: 没有加载到任何数据！")
        exit(1)

    image_paths  = np.array([item['image']  for item in train_data], dtype=object)
    coords_array = np.array([item['coords'] for item in train_data], dtype=np.float32)

    save_path = os.path.join(BASE_DIR, "train_data.npz")
    np.savez(save_path, image_paths=image_paths, coords=coords_array)

    print(f"\n成功加载 {len(train_data)} 张图片")
    print(f"坐标形状: {coords_array.shape}")
    print(f"坐标范围: min={coords_array.min():.4f}  max={coords_array.max():.4f}")
    print(f"  → 正常范围 [0, 1]，超出请检查 generate_labels.py")
    print(f"数据已保存: {save_path}")