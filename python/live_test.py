import torch
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp
import os
import time


def square_crop_frame(frame, x1, y1, x2, y2):
    crop_w, crop_h = x2-x1, y2-y1
    side = max(crop_w, crop_h)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    H, W = frame.shape[:2]
    sq_x1 = max(0, cx - side//2)
    sq_y1 = max(0, cy - side//2)
    sq_x2 = min(W, sq_x1 + side)
    sq_y2 = min(H, sq_y1 + side)
    crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]
    ah, aw = crop.shape[:2]
    if ah != aw:
        sq = np.zeros((side, side, 3), dtype=np.uint8)
        sq[:ah, :aw] = crop
        crop = sq
    return crop, sq_x1, sq_y1, side


# ★ 和 train_model_a.py 完全相同的模型定义
class HandNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v2
        backbone = mobilenet_v2(weights=None)
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone.last_channel, 42),
            nn.Sigmoid()          # ← 和训练一致，保留
        )
        self.model = backbone

    def forward(self, x): return self.model(x)


MODEL_PATH = "model_a_best.pth"
if not os.path.exists(MODEL_PATH):
    print(f"错误：找不到 '{MODEL_PATH}'"); exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = HandNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device); model.eval()
print(f"模型加载成功，使用 {device}")

mp_hands = mp.solutions.hands
detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

COLORS = [
    (128,128,128),
    (0,0,255),(0,0,200),(0,0,160),(0,0,120),
    (0,255,0),(0,200,0),(0,160,0),(0,120,0),
    (255,0,0),(200,0,0),(160,0,0),(120,0,0),
    (255,255,0),(200,200,0),(160,160,0),(120,120,0),
    (255,0,255),(200,0,200),(160,0,160),(120,0,120),
]
CONNECTIONS = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]

INPUT_SIZE, PAD = 224, 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0; fps = 0.0; last_time = time.time()
coords_global = None; bbox_global = None

print("实时手部关键点检测 | 按 ESC 退出")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    frame_count += 1

    if frame_count % 30 == 0:
        fps = 30.0 / (time.time() - last_time)
        last_time = time.time()

    result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        xs = [p.x*W for p in lm.landmark]
        ys = [p.y*H for p in lm.landmark]
        x1 = max(0,   int(min(xs))-PAD)
        y1 = max(0,   int(min(ys))-PAD)
        x2 = min(W-1, int(max(xs))+PAD)
        y2 = min(H-1, int(max(ys))+PAD)
        bbox_global = (x1,y1,x2,y2)

        if (x2-x1)<20 or (y2-y1)<20:
            coords_global = None; continue

        crop, sq_x1, sq_y1, sq_side = square_crop_frame(frame, x1, y1, x2, y2)
        inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (INPUT_SIZE,INPUT_SIZE))
        inp = inp.astype(np.float32)/255.0
        inp_t = torch.from_numpy(np.transpose(inp,(2,0,1))).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(inp_t).squeeze().cpu().numpy()  # Sigmoid保证[0,1]，无需clip

        # 映射回全图绝对像素
        coords_full = np.zeros(42, dtype=np.float32)
        for i in range(21):
            coords_full[i*2]   = np.clip(pred[i*2]   * sq_side + sq_x1, 0, W-1)
            coords_full[i*2+1] = np.clip(pred[i*2+1] * sq_side + sq_y1, 0, H-1)
        coords_global = coords_full
    else:
        bbox_global = None; coords_global = None

    if bbox_global:
        x1,y1,x2,y2 = bbox_global
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

    if coords_global is not None:
        pts = [(int(coords_global[i*2]), int(coords_global[i*2+1])) for i in range(21)]
        for finger in CONNECTIONS:
            for j in range(len(finger)-1):
                cv2.line(frame, pts[finger[j]], pts[finger[j+1]], COLORS[finger[j]], 2)
        for i,(px,py) in enumerate(pts):
            cv2.circle(frame,(px,py),5,COLORS[i],-1)
            cv2.putText(frame,str(i),(px+6,py),cv2.FONT_HERSHEY_SIMPLEX,0.35,COLORS[i],1)
        status, sc = "Hand Detected", (0,255,0)
    else:
        status, sc = "No Hand / Detecting...", (0,80,220)

    cv2.putText(frame,f"Hand Keypoints | FPS: {fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame,status,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,sc,2)
    cv2.imshow("Live Test", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows(); detector.close()
print("程序已退出")