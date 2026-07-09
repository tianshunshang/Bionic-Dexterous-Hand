import cv2
import mediapipe as mp
import json
import os
from glob import glob

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,   # 降低阈值，提高检测率
    min_tracking_confidence=0.5
)

IMG_DIR = r"E:\PythonProject\hand_images"
OUT_DIR = r"E:\PythonProject\labels"
os.makedirs(OUT_DIR, exist_ok=True)

image_files = sorted(glob(f"{IMG_DIR}/*.jpg") +
                     glob(f"{IMG_DIR}/*.jpeg") +
                     glob(f"{IMG_DIR}/*.png"))
print(f"找到 {len(image_files)} 张图片")

joint_names = [
    'wrist',
    'thumb1',  'thumb2',  'thumb3',  'thumb4',
    'index1',  'index2',  'index3',  'index4',
    'middle1', 'middle2', 'middle3', 'middle4',
    'ring1',   'ring2',   'ring3',   'ring4',
    'pinky1',  'pinky2',  'pinky3',  'pinky4',
]

success_count = 0
fail_count = 0
fail_list = []

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"✗ 无法读取: {os.path.basename(img_path)}")
        fail_count += 1
        fail_list.append(os.path.basename(img_path))
        continue

    h, w = img.shape[:2]

    # 如果图片太小，放大后再检测
    scale = 1.0
    detect_img = img
    if max(h, w) < 300:
        scale = 300 / max(h, w)
        detect_img = cv2.resize(img, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    img_relative_path = f"../hand_images/{os.path.basename(img_path)}"

    label_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": img_relative_path,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "coord_type": "pixel",   # prepare_data.py 读取此字段决定是否归一化
    }

    json_name = os.path.splitext(os.path.basename(img_path))[0] + '.json'
    json_path = os.path.join(OUT_DIR, json_name)

    if results.multi_hand_landmarks:
        for i, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            # 像素坐标（除以 scale 还原到原图尺寸）
            px = float(lm.x * detect_img.shape[1]) / scale
            py = float(lm.y * detect_img.shape[0]) / scale
            px = max(0.0, min(float(w), px))
            py = max(0.0, min(float(h), py))
            label_data["shapes"].append({
                "label": joint_names[i],
                "points": [[px, py]],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            })
        success_count += 1
        print(f"✓ {json_name}")
    else:
        fail_count += 1
        fail_list.append(os.path.basename(img_path))
        print(f"✗ 未检测到手: {os.path.basename(img_path)}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

hands.close()
print(f"\n完成！成功: {success_count}  失败: {fail_count}")
if fail_list:
    print("失败文件列表（需手动标注）:")
    for f in fail_list:
        print(f"  {f}")
print(f"标注保存在: {OUT_DIR}")