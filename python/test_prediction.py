import torch
import torch.nn as nn
import cv2
import numpy as np
import glob
import random
import os


class HandNet(nn.Module):
    def __init__(self, num_classes=42):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


model_path = "model_a_best.pth"
if not os.path.exists(model_path):
    print(f"错误：找不到模型文件 '{model_path}'")
    print("请先运行 train_model_a.py 训练模型")
    exit(1)

model = HandNet()
try:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
except Exception as e:
    print(f"模型加载失败: {e}")
    print("可能原因：模型结构与训练时不一致")
    exit(1)
model.eval()
print("模型加载成功！")

# ========== 测试图片 ==========
joint_names = ['wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4',
               'index1', 'index2', 'index3', 'index4',
               'middle1', 'middle2', 'middle3', 'middle4',
               'ring1', 'ring2', 'ring3', 'ring4',
               'pinky1', 'pinky2', 'pinky3', 'pinky4']

test_images = sorted(glob.glob("hand_images/*.jpg"))
if not test_images:
    print("没有找到测试图片！")
    exit()

print(f"共找到 {len(test_images)} 张测试图片")
print("操作：N=下一张  P=上一张  R=随机  ESC=退出")

current_idx = 0

while True:
    img_path = test_images[current_idx]
    img = cv2.imread(img_path)
    if img is None:
        print(f"读取失败: {img_path}")
        current_idx = (current_idx + 1) % len(test_images)
        continue

    h, w = img.shape[:2]

    # 预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0)

    # 预测
    with torch.no_grad():
        pred = model(img_tensor)

    coords = pred.squeeze().numpy()

    # 画点
    display = img.copy()
    colors = [(128, 128, 128), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]

    for i in range(21):
        x = int(coords[i * 2] * w)
        y = int(coords[i * 2 + 1] * h)
        color = colors[i // 5] if i > 0 else colors[0]

        cv2.circle(display, (x, y), 8, color, -1)
        cv2.putText(display, str(i), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 显示信息
    cv2.putText(display, f"{current_idx + 1}/{len(test_images)}: {os.path.basename(img_path)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, "N=Next P=Prev R=Random ESC=Exit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Keypoint Prediction", display)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('n') or key == ord('N'):
        current_idx = (current_idx + 1) % len(test_images)
    elif key == ord('p') or key == ord('P'):
        current_idx = (current_idx - 1) % len(test_images)
    elif key == ord('r') or key == ord('R'):
        current_idx = random.randint(0, len(test_images) - 1)
    elif key == 27:
        break

cv2.destroyAllWindows()
print("测试完成！")